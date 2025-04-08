# Comparative Analysis of Machine Learning Models for Sentiment Classification of Movie Reviews
# Based on project document by Harshit Bhardwaj, Tanush Singhal, and Komal Nagda

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
import gc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load the IMDB dataset from Hugging Face
print("Loading IMDB dataset...")
imdb_dataset = load_dataset("stanfordnlp/imdb")

# Convert the dataset to pandas DataFrame for easier manipulation
train_val_df = pd.DataFrame(imdb_dataset['train'])
test_df = pd.DataFrame(imdb_dataset['test'])

# Display dataset information
print(f"Initial training set size: {len(train_val_df)}")
print(f"Test set size: {len(test_df)}")
print(f"Label distribution in initial training set: {train_val_df['label'].value_counts().to_dict()}")

# Data preprocessing function
def preprocess_text(text):
    """Basic preprocessing for text data"""
    # Convert to lowercase
    text = text.lower()
    # You can add more preprocessing steps here as needed
    return text

# Apply preprocessing
train_val_df['processed_text'] = train_val_df['text'].apply(preprocess_text)
test_df['processed_text'] = test_df['text'].apply(preprocess_text)

# Sample a few reviews for inspection
print("\nSample positive review:")
print(train_val_df[train_val_df['label'] == 1]['text'].iloc[0][:300] + "...")
print("\nSample negative review:")
print(train_val_df[train_val_df['label'] == 0]['text'].iloc[0][:300] + "...")

# Create proper train/validation split with stratification to maintain class distribution
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.2,
    random_state=42,
    stratify=train_val_df['label']  # Ensure balanced class distribution
)

# Reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Verify the split worked correctly
print(f"\nAfter splitting:")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Training label distribution: {train_df['label'].value_counts().to_dict()}")
print(f"Validation label distribution: {val_df['label'].value_counts().to_dict()}")

# Extract features and labels for training and validation
train_texts = train_df['processed_text'].values
train_labels = train_df['label'].values
val_texts = val_df['processed_text'].values
val_labels = val_df['label'].values
test_texts = test_df['processed_text'].values
test_labels = test_df['label'].values

# Model 1: Traditional ML - Logistic Regression with TF-IDF n-grams
print("\n--- Model 1: Logistic Regression with TF-IDF n-grams ---")

# Start timing
start_time = time.time()

# Create TF-IDF vectorizer with n-grams
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
X_val_tfidf = tfidf_vectorizer.transform(val_texts)
X_test_tfidf = tfidf_vectorizer.transform(test_texts)

# Train logistic regression model
lr_model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='liblinear',
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train_tfidf, train_labels)

# Measure training time
lr_training_time = time.time() - start_time
print(f"Training time: {lr_training_time:.2f} seconds")

# Make predictions on validation and test sets
start_time = time.time()
val_preds_lr = lr_model.predict(X_val_tfidf)
test_preds_lr = lr_model.predict(X_test_tfidf)
lr_inference_time = time.time() - start_time
print(f"Inference time for {len(val_texts)} samples: {lr_inference_time:.2f} seconds")

# Calculate metrics
val_accuracy_lr = accuracy_score(val_labels, val_preds_lr)
val_precision_lr = precision_score(val_labels, val_preds_lr)
val_recall_lr = recall_score(val_labels, val_preds_lr)
val_f1_lr = f1_score(val_labels, val_preds_lr)

print(f"Validation Accuracy: {val_accuracy_lr:.4f}")
print(f"Validation Precision: {val_precision_lr:.4f}")
print(f"Validation Recall: {val_recall_lr:.4f}")
print(f"Validation F1-Score: {val_f1_lr:.4f}")

test_accuracy_lr = accuracy_score(test_labels, test_preds_lr)
test_precision_lr = precision_score(test_labels, test_preds_lr)
test_recall_lr = recall_score(test_labels, test_preds_lr)
test_f1_lr = f1_score(test_labels, test_preds_lr)

print(f"Test Accuracy: {test_accuracy_lr:.4f}")
print(f"Test Precision: {test_precision_lr:.4f}")
print(f"Test Recall: {test_recall_lr:.4f}")
print(f"Test F1-Score: {test_f1_lr:.4f}")

# Confusion matrix for logistic regression
cm_lr_val = confusion_matrix(val_labels, val_preds_lr)
cm_lr_test = confusion_matrix(test_labels, test_preds_lr)

# Model 2: DistilBERT Transformer Model
print("\n--- Model 2: DistilBERT Transformer Model ---")

try:
    # Check for GPU availability
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"GPU available. Using {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("GPU not available. Using CPU.")
        device = 'cpu'
    
    # Load DistilBERT tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    # Move model to device
    model.to(device)
    
    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(examples, padding="max_length", truncation=True, max_length=512)
    
    # Prepare datasets for DistilBERT
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
    
    # Tokenize datasets
    train_encodings = tokenize_function(train_texts.tolist())
    val_encodings = tokenize_function(val_texts.tolist())
    test_encodings = tokenize_function(test_texts.tolist())
    
    # Create dataset objects
    train_dataset = IMDBDataset(train_encodings, train_labels)
    val_dataset = IMDBDataset(val_encodings, val_labels)
    test_dataset = IMDBDataset(test_encodings, test_labels)
    
    # Define training arguments - modified for compatibility with different transformers versions
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,  # For demonstration, use fewer epochs
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save at the end of each epoch
        metric_for_best_model="f1",  # Use F1 score to determine best model
        load_best_model_at_end=True
    )
    
    # Create Trainer with evaluation metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    print("Training DistilBERT model...")
    start_time = time.time()
    trainer.train()
    distilbert_training_time = time.time() - start_time
    print(f"DistilBERT training time: {distilbert_training_time:.2f} seconds")
    
    # Evaluate on validation set
    start_time = time.time()
    val_results = trainer.evaluate(val_dataset)
    val_predictions = trainer.predict(val_dataset)
    distilbert_inference_time = time.time() - start_time
    print(f"DistilBERT inference time for {len(val_texts)} samples: {distilbert_inference_time:.2f} seconds")
    
    # Process predictions
    val_preds_bert = np.argmax(val_predictions.predictions, axis=1)
    val_accuracy_bert = accuracy_score(val_labels, val_preds_bert)
    val_precision_bert = precision_score(val_labels, val_preds_bert)
    val_recall_bert = recall_score(val_labels, val_preds_bert)
    val_f1_bert = f1_score(val_labels, val_preds_bert)
    
    print(f"Validation Accuracy: {val_accuracy_bert:.4f}")
    print(f"Validation Precision: {val_precision_bert:.4f}")
    print(f"Validation Recall: {val_recall_bert:.4f}")
    print(f"Validation F1-Score: {val_f1_bert:.4f}")
    
    # Evaluate on test set
    test_predictions = trainer.predict(test_dataset)
    test_preds_bert = np.argmax(test_predictions.predictions, axis=1)
    test_accuracy_bert = accuracy_score(test_labels, test_preds_bert)
    test_precision_bert = precision_score(test_labels, test_preds_bert)
    test_recall_bert = recall_score(test_labels, test_preds_bert)
    test_f1_bert = f1_score(test_labels, test_preds_bert)
    
    print(f"Test Accuracy: {test_accuracy_bert:.4f}")
    print(f"Test Precision: {test_precision_bert:.4f}")
    print(f"Test Recall: {test_recall_bert:.4f}")
    print(f"Test F1-Score: {test_f1_bert:.4f}")
    
    # Confusion matrix for DistilBERT
    cm_bert_val = confusion_matrix(val_labels, val_preds_bert)
    cm_bert_test = confusion_matrix(test_labels, test_preds_bert)
    
    # Clean up memory
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
except Exception as e:
    print(f"Error during DistilBERT training: {e}")
    # Provide fallback values to continue script execution
    val_accuracy_bert = 0
    val_precision_bert = 0
    val_recall_bert = 0
    val_f1_bert = 0
    test_accuracy_bert = 0
    test_precision_bert = 0
    test_recall_bert = 0
    test_f1_bert = 0
    distilbert_training_time = 0
    distilbert_inference_time = 0
    cm_bert_val = np.zeros((2, 2))
    cm_bert_test = np.zeros((2, 2))
    val_preds_bert = np.zeros(len(val_labels))

# Visualization of results
try:
    plt.figure(figsize=(20, 15))
    
    # Confusion Matrix for Logistic Regression - Validation
    plt.subplot(3, 3, 1)
    sns.heatmap(cm_lr_val, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Validation) - Logistic Regression')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Confusion Matrix for DistilBERT - Validation
    plt.subplot(3, 3, 2)
    sns.heatmap(cm_bert_val, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Validation) - DistilBERT')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Confusion Matrix for Logistic Regression - Test
    plt.subplot(3, 3, 4)
    sns.heatmap(cm_lr_test, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title('Confusion Matrix (Test) - Logistic Regression')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Confusion Matrix for DistilBERT - Test
    plt.subplot(3, 3, 5)
    sns.heatmap(cm_bert_test, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title('Confusion Matrix (Test) - DistilBERT')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Validation Metrics Comparison
    plt.subplot(3, 3, 3)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    lr_val_scores = [val_accuracy_lr, val_precision_lr, val_recall_lr, val_f1_lr]
    bert_val_scores = [val_accuracy_bert, val_precision_bert, val_recall_bert, val_f1_bert]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, lr_val_scores, width, label='Logistic Regression')
    plt.bar(x + width/2, bert_val_scores, width, label='DistilBERT')
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Validation Performance Metrics')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.legend()
    
    # Test Metrics Comparison
    plt.subplot(3, 3, 6)
    lr_test_scores = [test_accuracy_lr, test_precision_lr, test_recall_lr, test_f1_lr]
    bert_test_scores = [test_accuracy_bert, test_precision_bert, test_recall_bert, test_f1_bert]
    
    plt.bar(x - width/2, lr_test_scores, width, label='Logistic Regression')
    plt.bar(x + width/2, bert_test_scores, width, label='DistilBERT')
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Test Performance Metrics')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.legend()
    
    # Computational Efficiency
    plt.subplot(3, 3, 7)
    times = [lr_training_time, distilbert_training_time]
    labels = ['LR Training', 'BERT Training']
    plt.bar(labels, times)
    plt.yscale('log')  # Using log scale as the times may vary significantly
    plt.ylabel('Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    
    plt.subplot(3, 3, 8)
    inference_times = [lr_inference_time, distilbert_inference_time]
    inference_labels = ['LR Inference', 'BERT Inference']
    plt.bar(inference_labels, inference_times)
    plt.yscale('log')  # Using log scale as the times may vary significantly
    plt.ylabel('Time (seconds)')
    plt.title('Inference Time Comparison')
    plt.xticks(rotation=45)
    
    # Model efficiency vs performance scatter plot
    plt.subplot(3, 3, 9)
    plt.scatter([lr_training_time], [val_f1_lr], s=100, color='blue', label='LR (Val)')
    plt.scatter([distilbert_training_time], [val_f1_bert], s=100, color='red', label='BERT (Val)')
    plt.scatter([lr_training_time], [test_f1_lr], s=100, color='blue', marker='x', label='LR (Test)')
    plt.scatter([distilbert_training_time], [test_f1_bert], s=100, color='red', marker='x', label='BERT (Test)')
    plt.xscale('log')
    plt.xlabel('Training Time (log scale)')
    plt.ylabel('F1-Score')
    plt.title('Model Efficiency vs. Performance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png')
    print("Visualization saved as 'model_comparison_results.png'")
except Exception as e:
    print(f"Error creating visualization: {e}")

# Additional analysis: Sentiment Polarity Consistency Test
# For this test, we'll examine how each model performs on reviews with strong sentiment signals
print("\n--- Sentiment Polarity Consistency Test ---")

try:
    # Function to get prediction probabilities
    def get_prediction_probs(model, vectorizer, texts):
        X = vectorizer.transform(texts)
        return model.predict_proba(X)[:, 1]  # Probability of positive class
    
    # Generate sentiment consistency test with extreme reviews
    def create_extreme_reviews():
        extremely_positive = [
            "This movie is absolutely amazing! I loved every second of it. The acting, direction, and storyline were all perfect.",
            "One of the best films I've ever seen. I was blown away by the performances and cinematography.",
            "A masterpiece in every sense of the word. This film will be remembered as a classic for generations."
        ]
        
        extremely_negative = [
            "This movie was terrible. I hated everything about it and regret wasting my time watching it.",
            "Possibly the worst film I've ever seen. The acting was atrocious and the plot made no sense.",
            "A complete disaster from start to finish. I couldn't wait for it to end."
        ]
        
        return extremely_positive, extremely_negative
    
    extremely_positive, extremely_negative = create_extreme_reviews()
    
    # Logistic Regression predictions
    pos_probs_lr = get_prediction_probs(lr_model, tfidf_vectorizer, extremely_positive)
    neg_probs_lr = get_prediction_probs(lr_model, tfidf_vectorizer, extremely_negative)
    
    print("Logistic Regression results:")
    print(f"Average confidence for positive reviews: {np.mean(pos_probs_lr):.4f}")
    print(f"Average confidence for negative reviews: {np.mean(neg_probs_lr):.4f}")
    print(f"Polarity gap: {np.mean(pos_probs_lr) - np.mean(neg_probs_lr):.4f}")
    
    # Load a new DistilBERT model just for predictions
    # This is more efficient than loading the trained model again
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        
        # Try to load the best model if it exists
        try:
            model.load_state_dict(torch.load('./results/pytorch_model.bin'))
            print("Loaded saved model for predictions")
        except:
            print("Using pretrained model for predictions")
            
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)  # -1 for CPU
        
        def get_bert_sentiment_scores(texts):
            results = nlp(texts)
            scores = []
            for result in results:
                if result['label'] == 'LABEL_1':  # Positive
                    scores.append(result['score'])
                else:  # Negative
                    scores.append(1 - result['score'])
            return scores
        
        pos_scores_bert = get_bert_sentiment_scores(extremely_positive)
        neg_scores_bert = get_bert_sentiment_scores(extremely_negative)
        
        print("\nDistilBERT results:")
        print(f"Average confidence for positive reviews: {np.mean(pos_scores_bert):.4f}")
        print(f"Average confidence for negative reviews: {np.mean(neg_scores_bert):.4f}")
        print(f"Polarity gap: {np.mean(pos_scores_bert) - np.mean(neg_scores_bert):.4f}")
        
        # Clean up
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error in DistilBERT sentiment analysis: {e}")
except Exception as e:
    print(f"Error in sentiment polarity test: {e}")

# Conclusion and summary of findings
print("\n--- Conclusion ---")
print("Model comparison summary:")
print("Validation set results:")
print(f"Logistic Regression - Accuracy: {val_accuracy_lr:.4f}, Precision: {val_precision_lr:.4f}, Recall: {val_recall_lr:.4f}, F1-score: {val_f1_lr:.4f}")
print(f"DistilBERT - Accuracy: {val_accuracy_bert:.4f}, Precision: {val_precision_bert:.4f}, Recall: {val_recall_bert:.4f}, F1-score: {val_f1_bert:.4f}")

print("\nTest set results:")
print(f"Logistic Regression - Accuracy: {test_accuracy_lr:.4f}, Precision: {test_precision_lr:.4f}, Recall: {test_recall_lr:.4f}, F1-score: {test_f1_lr:.4f}")
print(f"DistilBERT - Accuracy: {test_accuracy_bert:.4f}, Precision: {test_precision_bert:.4f}, Recall: {test_recall_bert:.4f}, F1-score: {test_f1_bert:.4f}")

# Compare model performance
if test_f1_bert > test_f1_lr:
    better_model = "DistilBERT"
    improvement = (test_f1_bert - test_f1_lr) / test_f1_lr * 100
    print(f"\n{better_model} outperforms Logistic Regression by {improvement:.2f}% on test set F1-score.")
else:
    better_model = "Logistic Regression"
    improvement = (test_f1_lr - test_f1_bert) / test_f1_bert * 100
    print(f"\n{better_model} outperforms DistilBERT by {improvement:.2f}% on test set F1-score.")

print(f"\nComputational efficiency comparison:")
print(f"Logistic Regression - Training: {lr_training_time:.2f}s, Inference: {lr_inference_time:.2f}s")
print(f"DistilBERT - Training: {distilbert_training_time:.2f}s, Inference: {distilbert_inference_time:.2f}s")

if distilbert_training_time > 0:  # Only print ratio if DistilBERT training completed
    print(f"DistilBERT is {distilbert_training_time/lr_training_time:.1f}x slower in training and {distilbert_inference_time/lr_inference_time:.1f}x slower in inference.")

print("\nRecommendation based on findings:")
if test_f1_bert > test_f1_lr * 1.1:  # If BERT is at least 10% better
    print("DistilBERT provides significantly better performance and is recommended for applications where accuracy is critical and computational resources are available.")
elif test_f1_lr >= test_f1_bert * 0.9:  # If LR is within 10% of BERT
    print("Logistic Regression provides competitive performance with much better computational efficiency and is recommended for applications with limited resources or where speed is critical.")
else:
    print("Both models have their strengths. Choose DistilBERT for higher accuracy or Logistic Regression for faster processing and deployment.")