import torch
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, dataloader, device):
    true_labels = []
    predictions = []
    model.to(device)
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.view(-1).cpu().numpy())
            predictions.extend(preds.view(-1).cpu().numpy())
    end_time = time.time()
    total_time = end_time - start_time
    average_inference_time = total_time / len(dataloader.dataset)
    print(f"Average inference time per image: {average_inference_time:.4f} seconds.")
    conf_matrix = confusion_matrix(true_labels, predictions)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    print(classification_report(true_labels, predictions))

def plot_metrics(metrics_callback):
    epochs = np.arange(len(metrics_callback.train_losses)) + 1

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics_callback.train_losses, label='Train Loss')
    plt.plot(epochs, metrics_callback.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Losses')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics_callback.val_accs, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()