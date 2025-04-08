# sentiment_analysis_frontend.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
import pickle
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set page configuration
st.set_page_config(
    page_title="Movie Reviews Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for navigation and controls
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Data Exploration", "Model Training", "Model Comparison", "Test Your Own Review"])

# Add a progress placeholder to show during long operations
progress_placeholder = st.empty()

# Function to preprocess text
def preprocess_text(text):
    """Basic preprocessing for text data"""
    # Convert to lowercase
    text = text.lower()
    # You can add more preprocessing steps here as needed
    return text

# Custom loss tracker for Logistic Regression
class LogisticRegressionWithLossHistory:
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)
        self.train_losses = []
        self.val_losses = []
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=1000, epochs=10):
        # Initialize best model storage
        best_val_loss = float('inf')
        best_weights = None
        
        # Calculate number of batches
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # Training loop with epochs
        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_losses = []
            
            # Train in batches to simulate epochs
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Fit on this batch
                if epoch == 0 and i == 0:
                    # First batch of first epoch - initialize the model
                    self.model.fit(X_batch, y_batch)
                else:
                    # Subsequent batches - warm start from previous state
                    self.model.warm_start = True
                    self.model.fit(X_batch, y_batch)
                
                # Get probabilities for this batch
                y_proba = self.model.predict_proba(X_batch)
                # Calculate loss
                batch_loss = log_loss(y_batch, y_proba)
                epoch_losses.append(batch_loss)
            
            # Average epoch loss
            avg_epoch_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_epoch_loss)
            
            # Validate if validation data is provided
            if X_val is not None and y_val is not None:
                y_val_proba = self.model.predict_proba(X_val)
                val_loss = log_loss(y_val, y_val_proba)
                self.val_losses.append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = self.model.coef_.copy()
                
                # Progress feedback
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.4f}")
        
        # Restore best model if we have validation data
        if X_val is not None and y_val is not None and best_weights is not None:
            self.model.coef_ = best_weights
            
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_loss_history(self):
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }

# Custom callback for DistilBERT to track loss
class LossTrackingCallback(torch.nn.Module):
    def __init__(self, val_dataset=None):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.val_dataset = val_dataset
        self.step_count = 0
        self.epoch_losses = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        # Track training loss
        if "loss" in logs:
            self.epoch_losses.append(logs["loss"])
        
        # When epoch ends, calculate average loss
        if state.is_world_process_zero and "epoch" in logs:
            avg_loss = np.mean(self.epoch_losses)
            self.train_losses.append(avg_loss)
            self.epoch_losses = []  # Reset for next epoch
            
            # Validation after each epoch if validation dataset is provided
            if self.val_dataset is not None:
                val_loss = self.compute_validation_loss()
                self.val_losses.append(val_loss)
                self.step_count += 1
    
    def compute_validation_loss(self):
        model = self.model
        model.eval()  # Set model to evaluation mode
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(model.device)
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                count += 1
        
        model.train()  # Set model back to training mode
        return total_loss / count if count > 0 else 0
    
    def get_loss_history(self):
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }

# Function to train LR model
@st.cache_resource
def train_logistic_regression(train_texts, train_labels, val_texts, val_labels):
    """Train and evaluate Logistic Regression model with epoch-based training and loss tracking"""
    # Create progress bar
    progress_bar = progress_placeholder.progress(0)
    
    # Create TF-IDF vectorizer with n-grams
    progress_bar.progress(10)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
    X_val_tfidf = tfidf_vectorizer.transform(val_texts)
    
    progress_bar.progress(30)
    # Create custom LR model with loss tracking
    lr_model = LogisticRegressionWithLossHistory(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        max_iter=100,  # We'll handle iterations manually
        random_state=42
    )
    
    # Number of epochs for training
    epochs = 5
    
    progress_bar.progress(40)
    # Train the model with loss tracking
    start_time = time.time()
    lr_model.fit(
        X_train_tfidf, 
        train_labels, 
        X_val=X_val_tfidf, 
        y_val=val_labels,
        epochs=epochs,
        batch_size=1000
    )
    training_time = time.time() - start_time
    
    # Retrieve loss history
    loss_history = lr_model.get_loss_history()
    
    progress_bar.progress(70)
    # Make predictions on validation set
    start_time = time.time()
    val_preds_lr = lr_model.predict(X_val_tfidf)
    inference_time = time.time() - start_time
    
    progress_bar.progress(90)
    # Calculate metrics
    val_accuracy = accuracy_score(val_labels, val_preds_lr)
    val_precision = precision_score(val_labels, val_preds_lr)
    val_recall = recall_score(val_labels, val_preds_lr)
    val_f1 = f1_score(val_labels, val_preds_lr)
    
    # Create confusion matrix
    cm = confusion_matrix(val_labels, val_preds_lr)
    
    # Loss plot
    epochs_range = range(1, epochs+1)
    loss_fig = plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, loss_history['train_loss'], 'bo-', label='Training Loss')
    if loss_history['val_loss']:
        plt.plot(epochs_range, loss_history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss - Logistic Regression')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    progress_bar.progress(100)
    
    return lr_model.model, tfidf_vectorizer, val_preds_lr, val_accuracy, val_precision, val_recall, val_f1, cm, training_time, inference_time, loss_fig

# Custom dataset class for BERT
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# Function to train DistilBERT model
@st.cache_resource
def train_distilbert(train_texts, train_labels, val_texts, val_labels):
    """Train and evaluate DistilBERT model with loss tracking"""
    # Create progress bar
    progress_bar = progress_placeholder.progress(0)
    
    # Load DistilBERT tokenizer and model
    progress_bar.progress(10)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    progress_bar.progress(20)
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples, padding="max_length", truncation=True, max_length=512)
    
    progress_bar.progress(30)
    # Tokenize datasets
    train_encodings = tokenize_function(train_texts.tolist())
    val_encodings = tokenize_function(val_texts.tolist())
    
    # Create dataset objects
    train_dataset = IMDBDataset(train_encodings, train_labels)
    val_dataset = IMDBDataset(val_encodings, val_labels)
    
    # Create loss tracking callback
    loss_tracking_callback = LossTrackingCallback(val_dataset)
    
    progress_bar.progress(40)
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,  # Use 3 epochs for demonstration
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_steps=200,
        load_best_model_at_end=True,
        report_to="none"  # Disable wandb/tensorboard reporting
    )
    
    # Create dataloader for validation (needed for our custom callback)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False
    )
    loss_tracking_callback.val_dataloader = val_dataloader
    
    progress_bar.progress(50)
    # Create Trainer with custom callback
    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_history = {'train_loss': [], 'val_loss': []}
            
        def log(self, logs):
            # Save loss at each log step
            if "loss" in logs:
                self.loss_history['train_loss'].append(logs["loss"])
            super().log(logs)
            
        def evaluate(self, *args, **kwargs):
            # Run standard evaluation
            results = super().evaluate(*args, **kwargs)
            # Save validation loss
            if "eval_loss" in results:
                self.loss_history['val_loss'].append(results["eval_loss"])
            return results
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Connect model to callback
    loss_tracking_callback.model = model
    
    # Train the model
    progress_bar.progress(60)
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Plot loss curves
    train_loss = trainer.loss_history['train_loss']
    val_loss = trainer.loss_history['val_loss']
    
    # Extract losses at epoch level for plotting
    epoch_train_losses = []
    steps_per_epoch = len(train_dataset) // training_args.per_device_train_batch_size
    for i in range(int(training_args.num_train_epochs)):
        start_idx = i * steps_per_epoch
        end_idx = (i + 1) * steps_per_epoch
        if start_idx < len(train_loss):
            epoch_train_losses.append(np.mean(train_loss[start_idx:end_idx]))
    
    loss_fig = plt.figure(figsize=(10, 6))
    epochs = range(1, len(epoch_train_losses) + 1)
    plt.plot(epochs, epoch_train_losses, 'bo-', label='Training Loss')
    if val_loss:
        plt.plot(range(1, len(val_loss) + 1), val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss - DistilBERT')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    progress_bar.progress(80)
    # Evaluate on validation set
    start_time = time.time()
    predictions = trainer.predict(val_dataset)
    inference_time = time.time() - start_time
    
    # Process predictions
    preds = np.argmax(predictions.predictions, axis=1)
    val_accuracy = accuracy_score(val_labels, preds)
    val_precision = precision_score(val_labels, preds)
    val_recall = recall_score(val_labels, preds)
    val_f1 = f1_score(val_labels, preds)
    
    # Create confusion matrix
    cm = confusion_matrix(val_labels, preds)
    
    progress_bar.progress(100)
    
    return model, tokenizer, preds, val_accuracy, val_precision, val_recall, val_f1, cm, training_time, inference_time, loss_fig

# Main app functionality
@st.cache_data
def load_data():
    """Load and prepare IMDB dataset with train-validation split"""
    # Load the IMDB dataset from Hugging Face
    imdb_dataset = load_dataset("stanfordnlp/imdb")
    
    # Convert to pandas DataFrames
    train_df = pd.DataFrame(imdb_dataset['train'])
    test_df = pd.DataFrame(imdb_dataset['test'])
    
    # Preprocess text
    train_df['processed_text'] = train_df['text'].apply(preprocess_text)
    test_df['processed_text'] = test_df['text'].apply(preprocess_text)
    
    # Split into train/validation - use 80/20 split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['processed_text'].values, 
        train_df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=train_df['label'].values  # Ensure balanced classes in both splits
    )
    
    return train_df, test_df, train_texts, val_texts, train_labels, val_labels

# Function to visualize confusion matrix
def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    return fig

# Function to visualize metrics comparison
def plot_metrics_comparison(lr_metrics, bert_metrics):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, lr_metrics, width, label='Logistic Regression')
    ax.bar(x + width/2, bert_metrics, width, label='DistilBERT')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend()
    
    return fig

# Function to visualize computational efficiency
def plot_computational_efficiency(lr_times, bert_times):
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['Training Time', 'Inference Time']
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, lr_times, width, label='Logistic Regression')
    ax.bar(x + width/2, bert_times, width, label='DistilBERT')
    ax.set_xlabel('Operation')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computational Efficiency Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    return fig

# Introduction page
if page == "Introduction":
    st.title("🎬 Movie Reviews Sentiment Analysis")
    st.subheader("A Comparative Analysis of Machine Learning Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Project Overview
        This application demonstrates sentiment analysis on movie reviews using two different approaches:
        
        1. **Traditional ML**: Logistic Regression with TF-IDF n-grams
        2. **Deep Learning**: DistilBERT Transformer Model
        
        ### Features
        - Explore the IMDB dataset
        - Train and compare both models with train-validation split
        - Visualize loss curves during training
        - Compare performance metrics and confusion matrices
        - Test the models with your own movie reviews
        """)
    
    with col2:
        st.markdown("""
        ### How to Use This App
        1. **Data Exploration**: View sample reviews, dataset statistics, and the train-validation split
        2. **Model Training**: Train both models and monitor training/validation loss
        3. **Model Comparison**: Compare model performance with visualization
        4. **Test Your Own Review**: Enter your own movie review to see predictions
        
        ### Dataset
        We're using the IMDB movie reviews dataset from Hugging Face, which contains 50K reviews labeled as positive or negative. The data is split into 80% training and 20% validation.
        """)
    
    st.info("👈 Use the navigation panel on the left to explore different sections of the app.")

# Data Exploration page
elif page == "Data Exploration":
    st.title("📊 Data Exploration")
    
    # Load data
    with st.spinner("Loading IMDB dataset..."):
        train_df, test_df, train_texts, val_texts, train_labels, val_labels = load_data()
    
    # Display dataset information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Set Size", len(train_texts))
    with col2:
        st.metric("Validation Set Size", len(val_texts))
    with col3:
        st.metric("Test Set Size", len(test_df))
    
    # Class distribution
    st.subheader("Label Distribution")
    
    # Create figure with two subplots for training and validation sets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training set label distribution
    train_label_counts = np.bincount(train_labels)
    ax1.bar(['Negative (0)', 'Positive (1)'], [train_label_counts[0], train_label_counts[1]])
    ax1.set_ylabel('Count')
    ax1.set_title('Label Distribution in Training Set')
    
    # Validation set label distribution
    val_label_counts = np.bincount(val_labels)
    ax2.bar(['Negative (0)', 'Positive (1)'], [val_label_counts[0], val_label_counts[1]])
    ax2.set_ylabel('Count')
    ax2.set_title('Label Distribution in Validation Set')
    
    st.pyplot(fig)
    
    # Sample reviews
    st.subheader("Sample Reviews")
    tab1, tab2 = st.tabs(["Positive Reviews", "Negative Reviews"])
    
    with tab1:
        for i in range(3):
            sample = train_df[train_df['label'] == 1]['text'].iloc[i]
            st.text_area(f"Positive Sample {i+1}", sample[:300] + "...", height=100)
    
    with tab2:
        for i in range(3):
            sample = train_df[train_df['label'] == 0]['text'].iloc[i]
            st.text_area(f"Negative Sample {i+1}", sample[:300] + "...", height=100)
    
    # Review length distribution
    st.subheader("Review Length Distribution")
    train_df['review_length'] = train_df['text'].apply(len)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=train_df, x='review_length', hue='label', bins=30, kde=True, ax=ax)
    ax.set_title('Review Length Distribution by Sentiment')
    ax.set_xlabel('Review Length (characters)')
    st.pyplot(fig)
    
    # Train-validation split visualization
    st.subheader("Train-Validation Split")
    
    # Show statistics about the split
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Training Set Size", len(train_texts))
        st.metric("Training Positive Reviews", np.sum(train_labels == 1))
        st.metric("Training Negative Reviews", np.sum(train_labels == 0))
        train_pos_pct = np.sum(train_labels == 1) / len(train_labels) * 100
        train_neg_pct = np.sum(train_labels == 0) / len(train_labels) * 100
        st.text(f"Positive: {train_pos_pct:.1f}%, Negative: {train_neg_pct:.1f}%")
        
    with col2:
        st.metric("Validation Set Size", len(val_texts))
        st.metric("Validation Positive Reviews", np.sum(val_labels == 1))
        st.metric("Validation Negative Reviews", np.sum(val_labels == 0))
        val_pos_pct = np.sum(val_labels == 1) / len(val_labels) * 100
        val_neg_pct = np.sum(val_labels == 0) / len(val_labels) * 100
        st.text(f"Positive: {val_pos_pct:.1f}%, Negative: {val_neg_pct:.1f}%")
    
    # Visualize the split
    fig, ax = plt.subplots(figsize=(10, 6))
    splits = ['Training', 'Validation']
    sizes = [len(train_texts), len(val_texts)]
    ax.pie(sizes, labels=splits, autopct='%1.1f%%', startangle=90, explode=(0.05, 0))
    ax.axis('equal')
    ax.set_title('Train-Validation Split (80%-20%)')
    st.pyplot(fig)

# Model Training page
elif page == "Model Training":
    st.title("🔍 Model Training with Loss Visualization")
    
    # Load data
    with st.spinner("Loading IMDB dataset..."):
        train_df, test_df, train_texts, val_texts, train_labels, val_labels = load_data()
    
    st.info("Note: Training is cached after the first run to save time. To retrain, restart the app.")
    
    # Train models
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Logistic Regression (TF-IDF n-grams)")
        if st.button("Train Logistic Regression", key="train_lr"):
            with st.spinner("Training Logistic Regression model with train-validation split..."):
                lr_model, tfidf_vectorizer, lr_preds, lr_accuracy, lr_precision, lr_recall, lr_f1, lr_cm, lr_train_time, lr_infer_time, lr_loss_fig = train_logistic_regression(train_texts, train_labels, val_texts, val_labels)
                
                # Save model artifacts for later use
                with open('lr_model.pkl', 'wb') as f:
                    pickle.dump(lr_model, f)
                with open('tfidf_vectorizer.pkl', 'wb') as f:
                    pickle.dump(tfidf_vectorizer, f)
                
                # Display metrics
                st.metric("Accuracy", f"{lr_accuracy:.4f}")
                st.metric("F1-Score", f"{lr_f1:.4f}")
                st.metric("Training Time", f"{lr_train_time:.2f} seconds")
                st.metric("Inference Time", f"{lr_infer_time:.2f} seconds")
                
                # Show loss graph
                st.subheader("Training and Validation Loss")
                st.pyplot(lr_loss_fig)
                
                # Plot confusion matrix
                st.subheader("Confusion Matrix")
                st.pyplot(plot_confusion_matrix(lr_cm, "Logistic Regression Confusion Matrix"))
        
        else:
            if os.path.exists('lr_model.pkl'):
                st.success("Logistic Regression model already trained! Click the button to train again if needed.")
            else:
                st.warning("Click the button to train the Logistic Regression model")
    
    with col2:
        st.subheader("DistilBERT Transformer Model")
        if st.button("Train DistilBERT", key="train_bert"):
            with st.spinner("Training DistilBERT model with train-validation split... This might take 10-30 minutes"):
                bert_model, bert_tokenizer, bert_preds, bert_accuracy, bert_precision, bert_recall, bert_f1, bert_cm, bert_train_time, bert_infer_time, bert_loss_fig = train_distilbert(train_texts, train_labels, val_texts, val_labels)
                
                # Save model artifacts for later use
                with open('bert_metrics.pkl', 'wb') as f:
                    pickle.dump({
                        'accuracy': bert_accuracy,
                        'precision': bert_precision,
                        'recall': bert_recall,
                        'f1': bert_f1,
                        'cm': bert_cm,
                        'train_time': bert_train_time,
                        'infer_time': bert_infer_time
                    }, f)
                
                # Display metrics
                st.metric("Accuracy", f"{bert_accuracy:.4f}")
                st.metric("F1-Score", f"{bert_f1:.4f}")
                st.metric("Training Time", f"{bert_train_time:.2f} seconds")
                st.metric("Inference Time", f"{bert_infer_time:.2f} seconds")
                
                # Show loss graph
                st.subheader("Training and Validation Loss")
                st.pyplot(bert_loss_fig)
                
                # Plot confusion matrix
                st.subheader("Confusion Matrix")
                st.pyplot(plot_confusion_matrix(bert_cm, "DistilBERT Confusion Matrix"))
                
                # Free up memory
                del bert_model
                del bert_tokenizer
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        else:
            if os.path.exists('bert_metrics.pkl'):
                st.success("DistilBERT model already trained! Click the button to train again if needed.")
            else:
                st.warning("Click the button to train the DistilBERT model (This will take longer)")

    # Information about train-validation split monitoring
    st.subheader("About Train-Validation Split Monitoring")
    st.markdown("""
    We use a train-validation split to monitor model performance during training:
    
    - **80% Training / 20% Validation**: This split allows us to train on most of the data while having enough samples to validate performance.
    - **Loss Tracking**: Both models track training and validation loss across epochs to monitor learning progress.
    - **Early Stopping**: For DistilBERT, we use early stopping to prevent overfitting by monitoring validation loss.
    - **Best Model Selection**: The models save the best performing weights based on validation metrics.
    
    This approach allows us to detect overfitting (when training loss continues to decrease but validation loss increases) and underfitting (when both losses remain high).
    """)

# Model Comparison page
elif page == "Model Comparison":
    st.title("📈 Model Comparison")
    
    # Load data and model metrics if available
    with st.spinner("Loading model metrics..."):
        # Check if model metrics are available
        lr_metrics_available = os.path.exists('lr_model.pkl') and os.path.exists('tfidf_vectorizer.pkl')
        bert_metrics_available = os.path.exists('bert_metrics.pkl')
        
        if not lr_metrics_available or not bert_metrics_available:
            st.warning("Please train both models first in the 'Model Training' section.")
            st.stop()
        
        # Load metrics
        if bert_metrics_available:
            with open('bert_metrics.pkl', 'rb') as f:
                bert_metrics_dict = pickle.load(f)
                bert_accuracy = bert_metrics_dict['accuracy']
                bert_precision = bert_metrics_dict['precision']
                bert_recall = bert_metrics_dict['recall']
                bert_f1 = bert_metrics_dict['f1']
                bert_cm = bert_metrics_dict['cm']
                bert_train_time = bert_metrics_dict['train_time']
                bert_infer_time = bert_metrics_dict['infer_time']
        
        # Train Logistic Regression if needed
        if lr_metrics_available:
            # We need to load the train/validation data first
            train_df, test_df, train_texts, val_texts, train_labels, val_labels = load_data()
            # Train the model again to get the metrics
            lr_model, tfidf_vectorizer, lr_preds, lr_accuracy, lr_precision, lr_recall, lr_f1, lr_cm, lr_train_time, lr_infer_time, _ = train_logistic_regression(train_texts, train_labels, val_texts, val_labels)
    
    # Display comparison metrics
    st.subheader("Performance Metrics Comparison")
    
    # Side-by-side metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Logistic Regression")
        st.metric("Accuracy", f"{lr_accuracy:.4f}")
        st.metric("Precision", f"{lr_precision:.4f}")
        st.metric("Recall", f"{lr_recall:.4f}")
        st.metric("F1-Score", f"{lr_f1:.4f}")
        st.metric("Training Time", f"{lr_train_time:.2f} seconds")
        st.metric("Inference Time", f"{lr_infer_time:.2f} seconds")
    
    with col2:
        st.markdown("### DistilBERT")
        st.metric("Accuracy", f"{bert_accuracy:.4f}")
        st.metric("Precision", f"{bert_precision:.4f}")
        st.metric("Recall", f"{bert_recall:.4f}")
        st.metric("F1-Score", f"{bert_f1:.4f}")
        st.metric("Training Time", f"{bert_train_time:.2f} seconds")
        st.metric("Inference Time", f"{bert_infer_time:.2f} seconds")
    
    # Metrics visualization
    st.subheader("Metrics Visualization")
    
    # Create metrics comparison plot
    metrics_fig = plot_metrics_comparison(
        [lr_accuracy, lr_precision, lr_recall, lr_f1],
        [bert_accuracy, bert_precision, bert_recall, bert_f1]
    )
    st.pyplot(metrics_fig)
    
    # Computational efficiency
    st.subheader("Computational Efficiency")
    times_fig = plot_computational_efficiency(
        [lr_train_time, lr_infer_time],
        [bert_train_time, bert_infer_time]
    )
    st.pyplot(times_fig)
    
    # Confusion matrices
    st.subheader("Confusion Matrices")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Logistic Regression")
        st.pyplot(plot_confusion_matrix(lr_cm, "Logistic Regression"))
    
    with col2:
        st.markdown("### DistilBERT")
        st.pyplot(plot_confusion_matrix(bert_cm, "DistilBERT"))
    
    # Model analysis
    st.subheader("Analysis")
    
    # Calculate percentage improvements
    acc_improvement = (bert_accuracy - lr_accuracy) / lr_accuracy * 100
    f1_improvement = (bert_f1 - lr_f1) / lr_f1 * 100
    training_ratio = bert_train_time / lr_train_time if lr_train_time > 0 else 0
    inference_ratio = bert_infer_time / lr_infer_time if lr_infer_time > 0 else 0
    
    st.markdown(f"""
    ### Performance Analysis
    DistilBERT {'outperforms' if bert_f1 > lr_f1 else 'underperforms compared to'} Logistic Regression on F1-score 
    by {abs(f1_improvement):.2f}% and {'outperforms' if bert_accuracy > lr_accuracy else 'underperforms compared to'} on accuracy 
    by {abs(acc_improvement):.2f}%.
    
    ### Computational Efficiency Analysis
    - DistilBERT is approximately **{training_ratio:.1f}x slower** in training compared to Logistic Regression.
    - DistilBERT is approximately **{inference_ratio:.1f}x slower** in inference compared to Logistic Regression.
    
    ### Model Characteristics
    
    **Logistic Regression:**
    - Pros: Fast training and inference, low computational resource requirements, interpretable model.
    - Cons: Limited capacity to capture complex language patterns and context.
    - Best for: Applications requiring quick deployment, real-time processing, or where computational resources are limited.
    
    **DistilBERT:**
    - Pros: Better understanding of context and language nuances, higher accuracy potential.
    - Cons: Computationally expensive, slower training and inference.
    - Best for: Applications where accuracy is critical and computational resources are available.
    
    ### Recommendation
    """)
    
    if bert_f1 > lr_f1 * 1.1:  # If BERT is at least 10% better
        st.success("DistilBERT provides significantly better performance and is recommended for applications where accuracy is critical and computational resources are available.")
    elif lr_f1 >= bert_f1 * 0.9:  # If LR is within 10% of BERT
        st.success("Logistic Regression provides competitive performance with much better computational efficiency and is recommended for applications with limited resources or where speed is critical.")
    else:
        st.success("Both models have their strengths. Choose DistilBERT for higher accuracy or Logistic Regression for faster processing and deployment.")

# Test Your Own Review page
elif page == "Test Your Own Review":
    st.title("🔮 Test Your Own Movie Review")
    
    # Load models if available
    lr_available = os.path.exists('lr_model.pkl') and os.path.exists('tfidf_vectorizer.pkl')
    
    if not lr_available:
        st.warning("Please train the models first in the 'Model Training' section.")
        st.stop()
    
    # Load Logistic Regression model
    with open('lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    # Option to load DistilBERT
    use_distilbert = st.checkbox("Use DistilBERT model for prediction", value=False)
    
    if use_distilbert:
        with st.spinner("Loading DistilBERT model..."):
            try:
                # Load a fresh DistilBERT model for inference
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
                
                # Create sentiment analysis pipeline
                classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)
            except Exception as e:
                st.error(f"Error loading DistilBERT model: {e}")
                use_distilbert = False
    
    # User input
    user_review = st.text_area("Enter your movie review:", height=150, 
                              help="Write a movie review and we'll predict its sentiment.")
    
    # Sample reviews for quick testing
    st.markdown("### Or try a sample review:")
    
    sample_positive = "This movie was fantastic! The acting was superb, and the plot kept me engaged throughout. I would definitely recommend it to anyone looking for a good film."
    sample_negative = "What a waste of time. The plot was confusing, the acting was terrible, and I was bored throughout the entire film. I wouldn't recommend this movie to anyone."
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sample Positive Review"):
            user_review = sample_positive
            st.experimental_rerun()
    with col2:
        if st.button("Sample Negative Review"):
            user_review = sample_negative
            st.experimental_rerun()
    
    # Predict sentiment when input is provided
    if user_review:
        st.subheader("Sentiment Analysis Results")
        
        # Preprocess the input text
        processed_review = preprocess_text(user_review)
        
        # Logistic Regression prediction
        lr_prediction_start = time.time()
        lr_proba = lr_model.predict_proba(tfidf_vectorizer.transform([processed_review]))[0]
        lr_prediction = "Positive" if lr_proba[1] > 0.5 else "Negative"
        lr_confidence = lr_proba[1] if lr_prediction == "Positive" else lr_proba[0]
        lr_prediction_time = time.time() - lr_prediction_start
        
        # DistilBERT prediction if selected
        if use_distilbert:
            bert_prediction_start = time.time()
            result = classifier(user_review)[0]
            pos_score = next((score['score'] for score in result if score['label'] == 'LABEL_1'), 0)
            neg_score = next((score['score'] for score in result if score['label'] == 'LABEL_0'), 0)
            bert_prediction = "Positive" if pos_score > neg_score else "Negative"
            bert_confidence = pos_score if bert_prediction == "Positive" else neg_score
            bert_prediction_time = time.time() - bert_prediction_start
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Logistic Regression")
            st.write(f"Prediction: **{lr_prediction}**")
            st.write(f"Confidence: {lr_confidence:.4f}")
            st.write(f"Prediction time: {lr_prediction_time*1000:.2f} ms")
            
            # Visualization of LR confidence
            fig, ax = plt.subplots(figsize=(8, 2))
            color = 'green' if lr_prediction == "Positive" else 'red'
            ax.barh(['Sentiment'], [lr_confidence], color=color)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_title("Confidence Score")
            st.pyplot(fig)
        
        if use_distilbert:
            with col2:
                st.markdown("### DistilBERT")
                st.write(f"Prediction: **{bert_prediction}**")
                st.write(f"Confidence: {bert_confidence:.4f}")
                st.write(f"Prediction time: {bert_prediction_time*1000:.2f} ms")
                
                # Visualization of BERT confidence
                fig, ax = plt.subplots(figsize=(8, 2))
                color = 'green' if bert_prediction == "Positive" else 'red'
                ax.barh(['Sentiment'], [bert_confidence], color=color)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_title("Confidence Score")
                st.pyplot(fig)
                
                # Clean up
                del model
                del tokenizer
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Interpretation of the result
        st.subheader("Review Analysis")
        
        # Determine feature importance for Logistic Regression
        if lr_prediction == "Positive":
            st.markdown("### Key Positive Indicators")
            
            # Get feature importances from the LR model
            feature_names = tfidf_vectorizer.get_feature_names_out()
            coefficients = lr_model.coef_[0]
            
            # Create a DataFrame of features and coefficients
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': coefficients
            })
            
            # Sort by importance (coefficient value) for positive sentiment
            positive_features = feature_importance.sort_values('importance', ascending=False).head(10)
            
            # Display most influential positive words in the review
            influential_words = []
            for word in positive_features['feature']:
                if word in processed_review.lower().split() or f" {word}" in processed_review.lower():
                    influential_words.append(word)
            
            if influential_words:
                st.write("Words in your review that suggest positive sentiment:")
                st.write(", ".join(influential_words[:5]))
            else:
                st.write("No specific strongly positive words identified, but the overall language pattern suggests positive sentiment.")
                
        else:
            st.markdown("### Key Negative Indicators")
            
            # Get feature importances from the LR model
            feature_names = tfidf_vectorizer.get_feature_names_out()
            coefficients = lr_model.coef_[0]
            
            # Create a DataFrame of features and coefficients
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': coefficients
            })
            
            # Sort by importance (coefficient value) for negative sentiment (lowest coefficients)
            negative_features = feature_importance.sort_values('importance').head(10)
            
            # Display most influential negative words in the review
            influential_words = []
            for word in negative_features['feature']:
                if word in processed_review.lower().split() or f" {word}" in processed_review.lower():
                    influential_words.append(word)
            
            if influential_words:
                st.write("Words in your review that suggest negative sentiment:")
                st.write(", ".join(influential_words[:5]))
            else:
                st.write("No specific strongly negative words identified, but the overall language pattern suggests negative sentiment.")
        
        # General sentiment tips
        st.markdown("### Sentiment Tips")
        st.markdown("""
        **For positive reviews, consider including:**
        - Specific praise for performances, direction, or storytelling
        - Words like "excellent," "brilliant," "engaging," "recommend"
        - Comparisons to other well-regarded films
        
        **For negative reviews, common patterns include:**
        - Criticism of specific elements (acting, script, pacing)
        - Words like "disappointing," "boring," "waste," "terrible"
        - Mentions of walking out or not finishing the film
        """)

# Clear progress placeholder when not needed
progress_placeholder.empty()