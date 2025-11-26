import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MetricsCalculator:
    """Calculate and visualize classification metrics"""

    def __init__(self, class_names):
        self.class_names = class_names

    def calculate_metrics(self, y_true, y_pred):
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        return metrics

    def get_classification_report(self, y_true, y_pred):
        """Get detailed classification report"""
        return classification_report(y_true, y_pred, target_names=self.class_names)

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    def plot_metrics(self, metrics_dict, save_path=None):
        """Plot metrics"""
        metrics_names = list(metrics_dict.keys())
        metrics_values = list(metrics_dict.values())

        plt.figure(figsize=(10, 6))
        plt.bar(metrics_names, metrics_values)
        plt.ylim([0, 1])
        plt.ylabel('Score')
        plt.title('Classification Metrics')
        plt.xticks(rotation=45)

        for i, v in enumerate(metrics_values):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()