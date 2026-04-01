"""
Fake News Detection System using TF-IDF and Basic Text Mining
Based on the case study for multilingual social media content detection
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Tuple, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Text Processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Advanced text preprocessing for fake news detection
    Handles multilingual content and social media noise
    """
    
    def __init__(self, 
                 remove_urls=True, 
                 remove_hashtags=False,
                 expand_abbreviations=True,
                 remove_stopwords=False,
                 min_word_length=2,
                 max_word_length=20):
        
        self.remove_urls = remove_urls
        self.remove_hashtags = remove_hashtags
        self.expand_abbreviations = expand_abbreviations
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        
        # Common abbreviations dictionary
        self.abbreviations = {
            "u": "you", "ur": "your", "r": "are", "lol": "laughing out loud",
            "lmao": "laughing my ass off", "omg": "oh my god", "idk": "i don't know",
            "btw": "by the way", "fyi": "for your information", "tbh": "to be honest",
            "imo": "in my opinion", "smh": "shaking my head", "ngl": "not gonna lie",
            "iykyk": "if you know you know", "fr": "for real", "nvm": "never mind"
        }
        
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, str):
            X = [X]
        
        return [self._preprocess_text(text) for text in X]
    
    def _preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Handle hashtags
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        else:
            # Keep hashtag content but remove #
            text = re.sub(r'#(\w+)', r'\1', text)
        
        # Handle mentions
        text = re.sub(r'@\w+', '', text)
        
        # Expand abbreviations
        if self.expand_abbreviations:
            text = self._expand_abbreviations(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation except for important ones
        text = re.sub(r'[^\w\s\!?\.,;:]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Remove stopwords if enabled
            if self.remove_stopwords and token in self.stop_words:
                continue
            
            # Length filtering
            if len(token) < self.min_word_length or len(token) > self.max_word_length:
                continue
            
            # Remove pure numbers
            if token.isdigit():
                continue
            
            filtered_tokens.append(token)
        
        # Lemmatization
        filtered_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        
        return ' '.join(filtered_tokens)
    
        
    def _expand_abbreviations(self, text):
        """Expand common abbreviations"""
        words = text.split()
        expanded_words = []
        for word in words:
            if word in self.abbreviations:
                expanded_words.extend(self.abbreviations[word].split())
            else:
                expanded_words.append(word)
        return ' '.join(expanded_words)


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract various text-based features for fake news detection
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, str):
            X = [X]
        
        features = []
        for text in X:
            feat_dict = self._extract_features(text)
            features.append(list(feat_dict.values()))
        
        return np.array(features)
    
    def _extract_features(self, text):
        """Extract various linguistic and stylistic features"""
        if not isinstance(text, str):
            text = ""
        
        features = {}
        
        # Basic text statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        
        # Average word length
        words = text.split()
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Punctuation counts
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['period_count'] = text.count('.')
        features['comma_count'] = text.count(',')
        
        # Capitalization features
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['uppercase_word_count'] = sum(1 for word in words if word.isupper())
        
        # Digit features
        features['digit_count'] = sum(c.isdigit() for c in text)
        features['digit_ratio'] = features['digit_count'] / len(text) if text else 0
        
        # Special characters
        features['special_char_count'] = sum(not c.isalnum() and not c.isspace() for c in text)
        
        # Readability approximation (characters per word)
        features['avg_chars_per_word'] = len(text) / len(words) if words else 0
        
        return features


class FakeNewsDetector:
    """
    Main Fake News Detection System
    """
    
    def __init__(self, model_type='logistic_regression'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = TextFeatureExtractor()
        self.tfidf_vectorizer = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Model configurations
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'naive_bayes': MultinomialNB(),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42)
        }
    
    def _create_pipeline(self):
        """Create the complete processing pipeline"""
        
        # TF-IDF features
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        
        # Combined features
        combined_features = FeatureUnion([
            ('tfidf', tfidf),
            ('text_features', self.feature_extractor)
        ])
        
        # Complete pipeline
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('features', combined_features),
            ('classifier', self.models[self.model_type])
        ])
        
        return pipeline
    
    def fit(self, X, y):
        """Train the model"""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create and train pipeline
        self.model = self._create_pipeline()
        self.model.fit(X, y_encoded)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        probabilities = self.model.predict_proba(X)
        return probabilities
    
    def evaluate(self, X, y):
        """Comprehensive model evaluation"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Make predictions
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        # Encode true labels
        y_encoded = self.label_encoder.transform(y)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1_score': f1_score(y, y_pred, average='weighted'),
        }
        
        # Add ROC AUC for binary classification
        if len(self.label_encoder.classes_) == 2:
            metrics['roc_auc'] = roc_auc_score(y_encoded, y_pred_proba[:, 1])
        
        # Classification report
        metrics['classification_report'] = classification_report(y, y_pred)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y, y_pred)
        
        return metrics
    
    def get_feature_importance(self, feature_names=None, top_n=20):
        """Get feature importance for interpretability"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Get the classifier from pipeline
        classifier = self.model.named_steps['classifier']
        
        # Different methods for different models
        if hasattr(classifier, 'coef_'):  # Linear models
            importance = np.abs(classifier.coef_[0])
        elif hasattr(classifier, 'feature_importances_'):  # Tree-based models
            importance = classifier.feature_importances_
        else:
            return None
        
        # Get feature names
        feature_union = self.model.named_steps['features']
        tfidf_features = feature_union.transformer_list[0][1].get_feature_names_out()
        text_feature_names = ['char_count', 'word_count', 'sentence_count', 'avg_word_length',
                             'exclamation_count', 'question_count', 'period_count', 'comma_count',
                             'uppercase_ratio', 'uppercase_word_count', 'digit_count', 'digit_ratio',
                             'special_char_count', 'avg_chars_per_word']
        
        all_feature_names = list(tfidf_features) + text_feature_names
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)


# Example usage and demonstration
def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    
    # Sample fake news examples
    fake_news = [
        "BREAKING: Scientists discover miracle cure for all cancers! Big Pharma is hiding this from you!!!",
        "SHOCKING: Government officials caught in massive corruption scandal - mainstream media won't cover this!",
        "ALERT: Alien spacecraft found in Antarctica, authorities covering up the truth!",
        "EXCLUSIVE: Celebrity death hoax confirmed - they're actually alive and hiding!",
        "MIRACLE: This one weird trick melts belly fat overnight while you sleep! Doctors hate this!",
        "CONSPIRACY: Moon landing was faked! New evidence proves it never happened!",
        "SHOCKING: Popular food product causes instant weight gain - banned in other countries!",
        "BREAKING: Politician caught in scandal that will end their career tomorrow!",
        "ALERT: Your smartphone is spying on you right now - here's how to stop it!",
        "SHOCKING: Ancient civilization discovered with advanced technology beyond our understanding!"
    ]
    
    # Sample real news examples
    real_news = [
        "Local government announces new infrastructure project to improve city roads",
        "Scientists publish research on climate change effects in Nature journal",
        "Federal Reserve announces decision on interest rates following economic meeting",
        "Health officials provide update on vaccination campaign progress",
        "Technology company reports quarterly earnings and business outlook",
        "University research team develops new method for renewable energy storage",
        "World leaders discuss trade agreements at international summit",
        "Medical study shows promising results for new treatment approach",
        "Stock market shows mixed performance as investors react to economic data",
        "Environmental agency releases annual report on air quality improvements"
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': fake_news + real_news,
        'label': ['FAKE'] * len(fake_news) + ['REAL'] * len(real_news)
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def main():
    """Main demonstration function"""
    
    print("=" * 60)
    print("FAKE NEWS DETECTION SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Create sample dataset
    print("\n1. Creating sample dataset...")
    df = create_sample_dataset()
    print(f"Dataset created with {len(df)} samples")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Test different models
    models_to_test = ['logistic_regression', 'random_forest', 'naive_bayes', 'svm']
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*40}")
        print(f"Testing {model_name.replace('_', ' ').title()}")
        print(f"{'='*40}")
        
        # Create and train model
        detector = FakeNewsDetector(model_type=model_name)
        detector.fit(X_train, y_train)
        
        # Evaluate
        metrics = detector.evaluate(X_test, y_test)
        results[model_name] = metrics
        
        # Print results
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        # Feature importance (if available)
        importance = detector.get_feature_importance(top_n=10)
        if importance is not None:
            print(f"\nTop 10 Important Features:")
            for idx, row in importance.iterrows():
                print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    comparison_df = pd.DataFrame({
        model: {
            'Accuracy': results[model]['accuracy'],
            'Precision': results[model]['precision'],
            'Recall': results[model]['recall'],
            'F1-Score': results[model]['f1_score']
        }
        for model in models_to_test
    }).T
    
    print(comparison_df.round(4))
    
    # Best model selection
    best_model = comparison_df['F1-Score'].idxmax()
    print(f"\nBest performing model: {best_model.replace('_', ' ').title()}")
    print(f"Best F1-Score: {comparison_df.loc[best_model, 'F1-Score']:.4f}")
    
    return results


if __name__ == "__main__":
    results = main()
