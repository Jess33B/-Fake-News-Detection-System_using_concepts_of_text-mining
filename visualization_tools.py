"""
Visualization Tools for Fake News Detection System
Provides comprehensive visualization capabilities for model analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class FakeNewsVisualizer:
    """
    Comprehensive visualization toolkit for fake news detection analysis
    """
    
    def __init__(self, detector=None):
        self.detector = detector
        self.colors = {
            'fake': '#FF6B6B',
            'real': '#4ECDC4',
            'neutral': '#95E77E'
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix", figsize=(8, 6)):
        """Plot confusion matrix with heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_true, y_prob, title="ROC Curve", figsize=(8, 6)):
        """Plot ROC curve for binary classification"""
        if len(np.unique(y_true)) != 2:
            print("ROC curve is only available for binary classification")
            return
        
        # Convert labels to binary
        label_encoder = self.detector.label_encoder if self.detector else None
        if label_encoder:
            y_true_binary = label_encoder.transform(y_true)
        else:
            y_true_binary = (y_true == np.unique(y_true)[1]).astype(int)
        
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob[:, 1])
        auc_score = np.trapz(tpr, fpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_prob, title="Precision-Recall Curve", figsize=(8, 6)):
        """Plot Precision-Recall curve"""
        if len(np.unique(y_true)) != 2:
            print("Precision-Recall curve is only available for binary classification")
            return
        
        # Convert labels to binary
        label_encoder = self.detector.label_encoder if self.detector else None
        if label_encoder:
            y_true_binary = label_encoder.transform(y_true)
        else:
            y_true_binary = (y_true == np.unique(y_true)[1]).astype(int)
        
        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_prob[:, 1])
        pr_auc = np.trapz(precision, recall)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, importance_df, title="Top Feature Importance", figsize=(10, 8)):
        """Plot feature importance as horizontal bar chart"""
        if importance_df is None or importance_df.empty:
            print("No feature importance data available")
            return
        
        plt.figure(figsize=figsize)
        
        # Take top 20 features
        top_features = importance_df.head(20)
        
        # Create horizontal bar plot
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.tight_layout()
        plt.show()
    
    def plot_word_cloud(self, texts, label=None, title="Word Cloud", figsize=(12, 8)):
        """Generate word cloud for given texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(combined_text)
        
        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        if label:
            plt.title(f"{title} - {label}")
        else:
            plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_class_distribution(self, labels, title="Class Distribution", figsize=(8, 6)):
        """Plot class distribution as pie chart and bar chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Count values
        label_counts = pd.Series(labels).value_counts()
        
        # Pie chart
        colors = [self.colors.get(label.lower(), '#999999') for label in label_counts.index]
        ax1.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('Distribution (Pie Chart)')
        
        # Bar chart
        ax2.bar(label_counts.index, label_counts.values, color=colors)
        ax2.set_title('Distribution (Bar Chart)')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        
        # Add value labels on bars
        for i, v in enumerate(label_counts.values):
            ax2.text(i, v + 0.5, str(v), ha='center')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def plot_text_statistics(self, texts, labels, title="Text Statistics by Class", figsize=(15, 10)):
        """Plot various text statistics grouped by class"""
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(labels, str):
            labels = [labels]
        
        # Calculate statistics
        stats = []
        for text, label in zip(texts, labels):
            stats.append({
                'label': label,
                'char_count': len(text),
                'word_count': len(text.split()),
                'sentence_count': text.count('.') + text.count('!') + text.count('?'),
                'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
            })
        
        stats_df = pd.DataFrame(stats)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Plot each statistic
        stat_columns = ['char_count', 'word_count', 'sentence_count', 
                       'avg_word_length', 'exclamation_count', 'question_count']
        
        for i, col in enumerate(stat_columns):
            sns.boxplot(data=stats_df, x='label', y=col, ax=axes[i])
            axes[i].set_title(col.replace('_', ' ').title())
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
        return stats_df
    
    def plot_model_comparison(self, results_dict, title="Model Comparison", figsize=(12, 8)):
        """Compare multiple models using different metrics"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Prepare data
        comparison_data = []
        for model_name, metrics_dict in results_dict.items():
            for metric in metrics:
                if metric in metrics_dict:
                    comparison_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Score': metrics_dict[metric]
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create grouped bar plot
        plt.figure(figsize=figsize)
        sns.barplot(data=comparison_df, x='Model', y='Score', hue='Metric')
        plt.title(title)
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curve(self, train_sizes, train_scores, val_scores, title="Learning Curve", figsize=(10, 6)):
        """Plot learning curve showing model performance over training size"""
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=figsize)
        
        # Plot training scores
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        # Plot validation scores
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_interactive_dashboard(self, X_test, y_test, y_pred, y_prob=None):
        """Create interactive dashboard using Plotly"""
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'Class Distribution', 
                          'Prediction Confidence', 'Text Length Distribution'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "histogram"}, {"type": "box"}]]
        )
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig.add_trace(
            go.Heatmap(z=cm, colorscale='Blues', 
                      showscale=False, name='Confusion Matrix'),
            row=1, col=1
        )
        
        # Class Distribution
        label_counts = pd.Series(y_test).value_counts()
        fig.add_trace(
            go.Pie(labels=label_counts.index, values=label_counts.values,
                  name='True Distribution'),
            row=1, col=2
        )
        
        # Prediction Confidence (if probabilities available)
        if y_prob is not None:
            confidence_scores = np.max(y_prob, axis=1)
            fig.add_trace(
                go.Histogram(x=confidence_scores, name='Confidence Scores',
                           nbinsx=20),
                row=2, col=1
            )
        
        # Text Length Distribution
        text_lengths = [len(text.split()) for text in X_test]
        fig.add_trace(
            go.Box(x=y_test, y=text_lengths, name='Text Length by Class'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Fake News Detection Dashboard",
            showlegend=False
        )
        
        fig.show()
    
    def create_comprehensive_report(self, X_test, y_test, y_pred, y_prob=None, 
                                  model_name="Model", save_path=None):
        """Create a comprehensive visualization report"""
        print(f"Generating Comprehensive Report for {model_name}")
        print("=" * 60)
        
        # 1. Class Distribution
        self.plot_class_distribution(y_test, title=f"True Class Distribution - {model_name}")
        
        # 2. Confusion Matrix
        self.plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - {model_name}")
        
        # 3. ROC and PR Curves (for binary classification)
        if y_prob is not None and len(np.unique(y_test)) == 2:
            self.plot_roc_curve(y_test, y_prob, title=f"ROC Curve - {model_name}")
            self.plot_precision_recall_curve(y_test, y_prob, title=f"PR Curve - {model_name}")
        
        # 4. Text Statistics
        self.plot_text_statistics(X_test, y_test, title=f"Text Statistics - {model_name}")
        
        # 5. Word Clouds by Class
        if isinstance(X_test, list):
            test_texts = X_test
        else:
            test_texts = X_test.tolist()
        
        for label in np.unique(y_test):
            class_texts = [text for text, lbl in zip(test_texts, y_test) if lbl == label]
            if class_texts:
                self.plot_word_cloud(class_texts, label=label, 
                                   title=f"Word Cloud - {model_name}")
        
        print(f"Report generation completed for {model_name}")


# Utility functions for quick visualization
def quick_visualize_results(detector, X_test, y_test, model_name="Fake News Detector"):
    """Quick visualization of detector results"""
    visualizer = FakeNewsVisualizer(detector)
    
    # Get predictions
    y_pred = detector.predict(X_test)
    y_prob = detector.predict_proba(X_test)
    
    # Create comprehensive report
    visualizer.create_comprehensive_report(
        X_test, y_test, y_pred, y_prob, model_name
    )


def compare_models_visual(results_dict, save_plots=False):
    """Visual comparison of multiple models"""
    visualizer = FakeNewsVisualizer()
    
    # Model comparison plot
    visualizer.plot_model_comparison(results_dict, title="Model Performance Comparison")
    
    # Create summary table
    comparison_df = pd.DataFrame({
        model: {
            'Accuracy': results_dict[model]['accuracy'],
            'Precision': results_dict[model]['precision'],
            'Recall': results_dict[model]['recall'],
            'F1-Score': results_dict[model]['f1_score']
        }
        for model in results_dict.keys()
    }).T
    
    print("\nModel Comparison Summary:")
    print("=" * 50)
    print(comparison_df.round(4))
    
    return comparison_df
