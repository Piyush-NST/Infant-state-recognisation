import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os
import json

def evaluate_model(y_true, y_pred, model_name, class_names=None):
    """
    Calculate metrics and print them.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1_Score': float(f1)
    }
    
    print(f"--- {model_name} Evaluation ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    os.makedirs('outputs', exist_ok=True)
    
    # Save metrics
    with open(f"outputs/{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    if class_names is not None:
        plot_confusion_matrix(y_true, y_pred, class_names, model_name)
    else:
        plot_confusion_matrix(y_true, y_pred, [str(i) for i in set(y_true)], model_name)
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'outputs/{model_name}_confusion_matrix.png')
    plt.close()

def plot_training_history(history, model_name):
    """Plot DL training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{model_name} Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'outputs/{model_name}_history.png')
    plt.close()
