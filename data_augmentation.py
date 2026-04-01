"""
Data Augmentation and Handling for Fake News Detection
Addresses class imbalance and data scarcity issues
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple
import re
from collections import Counter
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek
import warnings
warnings.filterwarnings('ignore')


class FakeNewsDataAugmenter:
    """
    Advanced data augmentation techniques for fake news detection
    Handles multilingual content and preserves semantic meaning
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Synonym dictionary for text augmentation
        self.synonyms = {
            'good': ['excellent', 'great', 'wonderful', 'fantastic', 'amazing'],
            'bad': ['terrible', 'awful', 'horrible', 'dreadful', 'poor'],
            'big': ['large', 'huge', 'enormous', 'massive', 'giant'],
            'small': ['tiny', 'little', 'mini', 'compact', 'petite'],
            'fast': ['quick', 'rapid', 'swift', 'speedy', 'hasty'],
            'slow': ['sluggish', 'gradual', 'leisurely', 'unhurried', 'delayed'],
            'important': ['crucial', 'vital', 'essential', 'significant', 'critical'],
            'breaking': ['urgent', 'emergency', 'critical', 'alert', 'developing'],
            'shocking': ['surprising', 'stunning', 'astonishing', 'amazing', 'startling'],
            'exclusive': ['special', 'unique', 'rare', 'limited', 'premium'],
            'miracle': ['wonder', 'amazing', 'incredible', 'extraordinary', 'phenomenal'],
            'scandal': ['controversy', 'crisis', 'outrage', 'uproar', 'dispute'],
            'government': ['administration', 'authorities', 'officials', 'bureaucracy', 'state'],
            'scientists': ['researchers', 'experts', 'academics', 'scholars', 'investigators'],
            'doctors': ['physicians', 'medical experts', 'healthcare professionals', 'clinicians'],
            'celebrity': ['star', 'famous person', 'public figure', 'personality', 'icon'],
            'politician': ['lawmaker', 'representative', 'official', 'leader', 'statesman']
        }
        
        # Common sensational phrases
        self.sensational_phrases = [
            "BREAKING NEWS:",
            "SHOCKING DISCOVERY:",
            "EXCLUSIVE REPORT:",
            "MIRACLE CURE:",
            "ALERT:",
            "CONSPIRACY REVEALED:",
            "SCANDAL EXPOSED:",
            "SECRET REVEALED:",
            "HIDDEN TRUTH:",
            "COVER-UP EXPOSED:"
        ]
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace random words with their synonyms"""
        words = text.split()
        new_words = words.copy()
        
        # Get words that have synonyms
        replaceable_words = [word for word in words if word.lower() in self.synonyms]
        
        if replaceable_words:
            for _ in range(min(n, len(replaceable_words))):
                word_to_replace = random.choice(replaceable_words)
                synonyms = self.synonyms[word_to_replace.lower()]
                synonym = random.choice(synonyms)
                
                # Replace the word
                idx = new_words.index(word_to_replace)
                new_words[idx] = synonym
                
                # Remove from replaceable list to avoid duplicates
                replaceable_words.remove(word_to_replace)
        
        return ' '.join(new_words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Insert random sensational phrases"""
        words = text.split()
        
        for _ in range(n):
            phrase = random.choice(self.sensational_phrases)
            insert_idx = random.randint(0, len(words))
            words.insert(insert_idx, phrase)
        
        return ' '.join(words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swap two words"""
        words = text.split()
        
        if len(words) >= 2:
            for _ in range(n):
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words"""
        words = text.split()
        
        if len(words) == 1:
            return text
        
        new_words = [word for word in words if random.random() > p]
        
        # Ensure at least one word remains
        if not new_words:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def augment_text(self, text: str, num_augmented: int = 3) -> List[str]:
        """Generate multiple augmented versions of a text"""
        augmented_texts = []
        
        augmentation_methods = [
            lambda x: self.synonym_replacement(x, n=1),
            lambda x: self.random_insertion(x, n=1),
            lambda x: self.random_swap(x, n=1),
            lambda x: self.random_deletion(x, p=0.1)
        ]
        
        for _ in range(num_augmented):
            augmented_text = text
            # Apply random augmentation
            method = random.choice(augmentation_methods)
            try:
                augmented_text = method(augmented_text)
                augmented_texts.append(augmented_text)
            except:
                augmented_texts.append(text)  # Fallback to original
        
        return augmented_texts
    
    def augment_dataset(self, df: pd.DataFrame, text_column: str, 
                       label_column: str, target_samples_per_class: int = None) -> pd.DataFrame:
        """Augment dataset to balance classes or reach target size"""
        augmented_data = []
        
        for label in df[label_column].unique():
            class_data = df[df[label_column] == label]
            current_count = len(class_data)
            
            # Determine how many samples to generate
            if target_samples_per_class:
                samples_needed = max(0, target_samples_per_class - current_count)
            else:
                # Balance with the majority class
                max_class_size = df[label_column].value_counts().max()
                samples_needed = max(0, max_class_size - current_count)
            
            # Add original data
            augmented_data.append(class_data)
            
            # Generate augmented samples if needed
            if samples_needed > 0:
                print(f"Generating {samples_needed} augmented samples for class '{label}'")
                
                augmented_samples = []
                texts = class_data[text_column].tolist()
                
                # Generate augmented texts
                for i in range(samples_needed):
                    original_text = texts[i % len(texts)]
                    augmented_texts = self.augment_text(original_text, num_augmented=1)
                    
                    for aug_text in augmented_texts:
                        if len(augmented_samples) < samples_needed:
                            augmented_samples.append({
                                text_column: aug_text,
                                label_column: label
                            })
                
                if augmented_samples:
                    augmented_df = pd.DataFrame(augmented_samples)
                    augmented_data.append(augmented_df)
        
        # Combine all data
        final_df = pd.concat(augmented_data, ignore_index=True)
        
        # Shuffle the dataset
        final_df = final_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return final_df


class ClassImbalanceHandler:
    """
    Handle class imbalance using various sampling techniques
    """
    
    def __init__(self):
        self.sampling_methods = {
            'oversample': self._oversample,
            'undersample': self._undersample,
            'smote': self._smote_sampling,
            'adasyn': self._adasyn_sampling,
            'smote_enn': self._smote_enn_sampling,
            'smote_tomek': self._smote_tomek_sampling,
            'nearmiss': self._nearmiss_sampling
        }
    
    def handle_imbalance(self, X, y, method='smote', **kwargs):
        """Apply specified sampling method to handle class imbalance"""
        if method not in self.sampling_methods:
            raise ValueError(f"Method '{method}' not supported. Available methods: {list(self.sampling_methods.keys())}")
        
        return self.sampling_methods[method](X, y, **kwargs)
    
    def _oversample(self, X, y, **kwargs):
        """Random oversampling of minority class"""
        sampler = RandomUnderSampler(sampling_strategy='auto', **kwargs)
        return sampler.fit_resample(X, y)
    
    def _undersample(self, X, y, **kwargs):
        """Random undersampling of majority class"""
        sampler = RandomUnderSampler(sampling_strategy='auto', **kwargs)
        return sampler.fit_resample(X, y)
    
    def _smote_sampling(self, X, y, **kwargs):
        """SMOTE oversampling"""
        sampler = SMOTE(sampling_strategy='auto', random_state=42, **kwargs)
        return sampler.fit_resample(X, y)
    
    def _adasyn_sampling(self, X, y, **kwargs):
        """ADASYN oversampling"""
        sampler = ADASYN(sampling_strategy='auto', random_state=42, **kwargs)
        return sampler.fit_resample(X, y)
    
    def _smote_enn_sampling(self, X, y, **kwargs):
        """Combined SMOTE and Edited Nearest Neighbors"""
        sampler = SMOTEENN(sampling_strategy='auto', random_state=42, **kwargs)
        return sampler.fit_resample(X, y)
    
    def _smote_tomek_sampling(self, X, y, **kwargs):
        """Combined SMOTE and Tomek Links"""
        sampler = SMOTETomek(sampling_strategy='auto', random_state=42, **kwargs)
        return sampler.fit_resample(X, y)
    
    def _nearmiss_sampling(self, X, y, **kwargs):
        """NearMiss undersampling"""
        sampler = NearMiss(sampling_strategy='auto', **kwargs)
        return sampler.fit_resample(X, y)
    
    def compare_sampling_methods(self, X, y, methods=['smote', 'adasyn', 'smote_enn']):
        """Compare different sampling methods"""
        results = {}
        
        original_counts = pd.Series(y).value_counts()
        print(f"Original class distribution: {original_counts.to_dict()}")
        
        for method in methods:
            try:
                X_resampled, y_resampled = self.handle_imbalance(X, y, method=method)
                resampled_counts = pd.Series(y_resampled).value_counts()
                results[method] = resampled_counts.to_dict()
                print(f"{method.upper()}: {resampled_counts.to_dict()}")
            except Exception as e:
                print(f"Error with {method}: {str(e)}")
        
        return results


class DataValidator:
    """
    Validate and clean dataset for fake news detection
    """
    
    def __init__(self):
        self.validation_rules = {
            'min_text_length': 10,
            'max_text_length': 10000,
            'min_words': 3,
            'max_words': 1000,
            'allowed_languages': ['en', 'es', 'fr', 'de', 'it', 'pt']  # Common languages
        }
    
    def validate_dataset(self, df: pd.DataFrame, text_column: str, 
                        label_column: str) -> Tuple[pd.DataFrame, Dict]:
        """Validate dataset and return cleaned data with validation report"""
        validation_report = {
            'original_shape': df.shape,
            'issues_found': [],
            'cleaned_shape': None,
            'quality_score': 0
        }
        
        # Make a copy
        cleaned_df = df.copy()
        
        # Check for missing values
        missing_text = cleaned_df[text_column].isnull().sum()
        missing_labels = cleaned_df[label_column].isnull().sum()
        
        if missing_text > 0:
            validation_report['issues_found'].append(f"Missing text values: {missing_text}")
            cleaned_df = cleaned_df.dropna(subset=[text_column])
        
        if missing_labels > 0:
            validation_report['issues_found'].append(f"Missing label values: {missing_labels}")
            cleaned_df = cleaned_df.dropna(subset=[label_column])
        
        # Check text length
        too_short = cleaned_df[text_column].str.len() < self.validation_rules['min_text_length']
        too_long = cleaned_df[text_column].str.len() > self.validation_rules['max_text_length']
        
        if too_short.sum() > 0:
            validation_report['issues_found'].append(f"Texts too short: {too_short.sum()}")
            cleaned_df = cleaned_df[~too_short]
        
        if too_long.sum() > 0:
            validation_report['issues_found'].append(f"Texts too long: {too_long.sum()}")
            cleaned_df = cleaned_df[~too_long]
        
        # Check word count
        word_counts = cleaned_df[text_column].str.split().str.len()
        too_few_words = word_counts < self.validation_rules['min_words']
        too_many_words = word_counts > self.validation_rules['max_words']
        
        if too_few_words.sum() > 0:
            validation_report['issues_found'].append(f"Texts with too few words: {too_few_words.sum()}")
            cleaned_df = cleaned_df[~too_few_words]
        
        if too_many_words.sum() > 0:
            validation_report['issues_found'].append(f"Texts with too many words: {too_many_words.sum()}")
            cleaned_df = cleaned_df[~too_many_words]
        
        # Check for duplicates
        duplicates = cleaned_df.duplicated(subset=[text_column]).sum()
        if duplicates > 0:
            validation_report['issues_found'].append(f"Duplicate texts: {duplicates}")
            cleaned_df = cleaned_df.drop_duplicates(subset=[text_column])
        
        # Check class distribution
        class_counts = cleaned_df[label_column].value_counts()
        min_class_size = class_counts.min()
        max_class_size = class_counts.max()
        imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
        
        if imbalance_ratio > 10:
            validation_report['issues_found'].append(f"High class imbalance: {imbalance_ratio:.2f}:1")
        
        # Calculate quality score
        total_issues = len(validation_report['issues_found'])
        validation_report['quality_score'] = max(0, 100 - (total_issues * 10))
        validation_report['cleaned_shape'] = cleaned_df.shape
        validation_report['class_distribution'] = class_counts.to_dict()
        validation_report['imbalance_ratio'] = imbalance_ratio
        
        return cleaned_df, validation_report
    
    def generate_quality_report(self, validation_report: Dict) -> str:
        """Generate a human-readable quality report"""
        report = []
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 40)
        report.append(f"Original dataset shape: {validation_report['original_shape']}")
        report.append(f"Cleaned dataset shape: {validation_report['cleaned_shape']}")
        report.append(f"Quality score: {validation_report['quality_score']}/100")
        report.append("")
        
        if validation_report['issues_found']:
            report.append("ISSUES FOUND:")
            for issue in validation_report['issues_found']:
                report.append(f"  - {issue}")
        else:
            report.append("No issues found!")
        
        report.append("")
        report.append("CLASS DISTRIBUTION:")
        for label, count in validation_report['class_distribution'].items():
            report.append(f"  {label}: {count}")
        
        report.append(f"Imbalance ratio: {validation_report['imbalance_ratio']:.2f}:1")
        
        return "\n".join(report)


# Utility functions
def create_balanced_dataset(df: pd.DataFrame, text_column: str, 
                           label_column: str, method='augmentation') -> pd.DataFrame:
    """Create a balanced dataset using specified method"""
    
    if method == 'augmentation':
        augmenter = FakeNewsDataAugmenter()
        max_class_size = df[label_column].value_counts().max()
        return augmenter.augment_dataset(df, text_column, label_column, max_class_size)
    
    elif method == 'undersampling':
        # Simple undersampling
        min_class_size = df[label_column].value_counts().min()
        balanced_dfs = []
        
        for label in df[label_column].unique():
            class_df = df[df[label_column] == label]
            sampled_df = class_df.sample(n=min_class_size, random_state=42)
            balanced_dfs.append(sampled_df)
        
        return pd.concat(balanced_dfs, ignore_index=True)
    
    else:
        raise ValueError(f"Method '{method}' not supported. Use 'augmentation' or 'undersampling'")


def analyze_dataset_characteristics(df: pd.DataFrame, text_column: str, 
                                  label_column: str) -> Dict:
    """Analyze dataset characteristics"""
    
    analysis = {}
    
    # Basic statistics
    analysis['total_samples'] = len(df)
    analysis['unique_texts'] = df[text_column].nunique()
    analysis['class_distribution'] = df[label_column].value_counts().to_dict()
    
    # Text statistics
    df['text_length'] = df[text_column].str.len()
    df['word_count'] = df[text_column].str.split().str.len()
    
    analysis['text_stats'] = {
        'avg_length': df['text_length'].mean(),
        'std_length': df['text_length'].std(),
        'min_length': df['text_length'].min(),
        'max_length': df['text_length'].max(),
        'avg_words': df['word_count'].mean(),
        'std_words': df['word_count'].std(),
        'min_words': df['word_count'].min(),
        'max_words': df['word_count'].max()
    }
    
    # Class imbalance
    class_counts = df[label_column].value_counts()
    min_class = class_counts.min()
    max_class = class_counts.max()
    analysis['imbalance_ratio'] = max_class / min_class if min_class > 0 else float('inf')
    
    # Missing values
    analysis['missing_values'] = {
        'text': df[text_column].isnull().sum(),
        'labels': df[label_column].isnull().sum()
    }
    
    # Duplicates
    analysis['duplicates'] = df.duplicated(subset=[text_column]).sum()
    
    return analysis


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'text': [
            "Breaking news: Scientists discover miracle cure!",
            "Government announces new policy changes",
            "SHOCKING: Celebrity caught in scandal!",
            "Local business reports quarterly earnings",
            "EXCLUSIVE: Alien spacecraft found!"
        ] * 10,
        'label': ['FAKE', 'REAL', 'FAKE', 'REAL', 'FAKE'] * 10
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original dataset:")
    print(df['label'].value_counts())
    
    # Data validation
    validator = DataValidator()
    cleaned_df, validation_report = validator.validate_dataset(df, 'text', 'label')
    
    print("\nValidation Report:")
    print(validator.generate_quality_report(validation_report))
    
    # Data augmentation
    augmenter = FakeNewsDataAugmenter()
    balanced_df = augmenter.augment_dataset(df, 'text', 'label', target_samples_per_class=20)
    
    print(f"\nAfter augmentation:")
    print(balanced_df['label'].value_counts())
    
    # Class imbalance handling
    imbalance_handler = ClassImbalanceHandler()
    
    # This would require feature extraction first
    # X_resampled, y_resampled = imbalance_handler.handle_imbalance(X, y, method='smote')
