"""
Complete Demo of Fake News Detection System
Demonstrates all components working together
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import our modules
from fake_news_detector import FakeNewsDetector, create_sample_dataset
from visualization_tools import FakeNewsVisualizer, quick_visualize_results, compare_models_visual
from data_augmentation import FakeNewsDataAugmenter, DataValidator, ClassImbalanceHandler

def comprehensive_demo():
    """Complete demonstration of the fake news detection system"""
    
    print("=" * 80)
    print("COMPREHENSIVE FAKE NEWS DETECTION SYSTEM DEMO")
    print("=" * 80)
    
    # Step 1: Create and validate dataset
    print("\n1. DATASET CREATION AND VALIDATION")
    print("-" * 50)
    
    # Create sample dataset
    df = create_sample_dataset()
    print(f"Created dataset with {len(df)} samples")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Validate dataset
    validator = DataValidator()
    cleaned_df, validation_report = validator.validate_dataset(df, 'text', 'label')
    
    print("\nValidation Report:")
    print(validator.generate_quality_report(validation_report))
    
    # Step 2: Data augmentation (if needed)
    print("\n2. DATA AUGMENTATION")
    print("-" * 50)
    
    # Check if we need augmentation
    imbalance_ratio = validation_report['imbalance_ratio']
    if imbalance_ratio > 2.0:
        print(f"Class imbalance detected (ratio: {imbalance_ratio:.2f}:1)")
        print("Applying data augmentation...")
        
        augmenter = FakeNewsDataAugmenter()
        max_class_size = cleaned_df['label'].value_counts().max()
        augmented_df = augmenter.augment_dataset(
            cleaned_df, 'text', 'label', max_class_size
        )
        
        print(f"After augmentation: {augmented_df['label'].value_counts().to_dict()}")
        df_to_use = augmented_df
    else:
        print("Dataset is reasonably balanced. No augmentation needed.")
        df_to_use = cleaned_df
    
    # Step 3: Split data
    print("\n3. DATA SPLITTING")
    print("-" * 50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_to_use['text'], df_to_use['label'], 
        test_size=0.3, random_state=42, stratify=df_to_use['label']
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Training distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Test distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    # Step 4: Model training and evaluation
    print("\n4. MODEL TRAINING AND EVALUATION")
    print("-" * 50)
    
    models_to_test = ['logistic_regression', 'random_forest', 'naive_bayes', 'svm']
    results = {}
    trained_models = {}
    
    for model_name in models_to_test:
        print(f"\nTraining {model_name.replace('_', ' ').title()}...")
        
        # Create and train detector
        detector = FakeNewsDetector(model_type=model_name)
        detector.fit(X_train, y_train)
        
        # Evaluate
        metrics = detector.evaluate(X_test, y_test)
        results[model_name] = metrics
        trained_models[model_name] = detector
        
        # Print key metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Step 5: Model comparison
    print("\n5. MODEL COMPARISON")
    print("-" * 50)
    
    comparison_df = compare_models_visual(results)
    
    # Select best model
    best_model_name = comparison_df['F1-Score'].idxmax()
    best_detector = trained_models[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\nBest model: {best_model_name.replace('_', ' ').title()}")
    print(f"Best F1-Score: {best_metrics['f1_score']:.4f}")
    
    # Step 6: Detailed analysis of best model
    print("\n6. DETAILED ANALYSIS OF BEST MODEL")
    print("-" * 50)
    
    # Get predictions and probabilities
    y_pred = best_detector.predict(X_test)
    y_prob = best_detector.predict_proba(X_test)
    
    # Feature importance
    print("\nFeature Importance:")
    importance_df = best_detector.get_feature_importance(top_n=15)
    if importance_df is not None:
        for idx, row in importance_df.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Step 7: Visualization
    print("\n7. VISUALIZATION")
    print("-" * 50)
    
    visualizer = FakeNewsVisualizer(best_detector)
    
    # Create comprehensive report
    visualizer.create_comprehensive_report(
        X_test, y_test, y_pred, y_prob, 
        f"Best Model: {best_model_name.replace('_', ' ').title()}"
    )
    
    # Step 8: Real-time prediction demo
    print("\n8. REAL-TIME PREDICTION DEMO")
    print("-" * 50)
    
    test_texts = [
        "BREAKING: Scientists discover miracle cure for cancer in ancient herbs!",
        "Federal Reserve announces decision on interest rates following economic meeting",
        "SHOCKING: Government conspiracy revealed in leaked documents!",
        "Local company reports quarterly earnings with modest growth",
        "EXCLUSIVE: Alien spacecraft found in Antarctica, authorities covering up truth!"
    ]
    
    print("Testing with sample texts:")
    for i, text in enumerate(test_texts, 1):
        prediction = best_detector.predict([text])[0]
        probabilities = best_detector.predict_proba([text])[0]
        confidence = np.max(probabilities)
        
        print(f"\nTest {i}:")
        print(f"Text: {text}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        
        # Show probability distribution
        for j, (label, prob) in enumerate(zip(best_detector.label_encoder.classes_, probabilities)):
            print(f"  P({label}): {prob:.4f}")
    
    # Step 9: Error analysis
    print("\n9. ERROR ANALYSIS")
    print("-" * 50)
    
    # Find misclassified examples
    misclassified_indices = np.where(y_test != y_pred)[0]
    
    if len(misclassified_indices) > 0:
        print(f"Found {len(misclassified_indices)} misclassified examples:")
        
        for idx in misclassified_indices[:5]:  # Show first 5
            print(f"\nExample {idx}:")
            print(f"True: {y_test.iloc[idx]}")
            print(f"Predicted: {y_pred[idx]}")
            print(f"Text: {X_test.iloc[idx]}")
            
            # Show prediction probabilities
            probs = y_prob[idx]
            for j, (label, prob) in enumerate(zip(best_detector.label_encoder.classes_, probs)):
                print(f"  P({label}): {prob:.4f}")
    else:
        print("No misclassified examples found!")
    
    # Step 10: Summary and recommendations
    print("\n10. SUMMARY AND RECOMMENDATIONS")
    print("-" * 50)
    
    print(f"System Performance Summary:")
    print(f"  Best Model: {best_model_name.replace('_', ' ').title()}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {best_metrics['f1_score']:.4f}")
    
    if 'roc_auc' in best_metrics:
        print(f"  ROC-AUC: {best_metrics['roc_auc']:.4f}")
    
    print(f"\nRecommendations:")
    print(f"  1. Consider collecting more diverse training data")
    print(f"  2. Implement cross-validation for more robust evaluation")
    print(f"  3. Add multilingual support for better coverage")
    print(f"  4. Consider ensemble methods for improved performance")
    print(f"  5. Implement real-time monitoring for concept drift")
    
    return {
        'results': results,
        'best_model': best_detector,
        'best_model_name': best_model_name,
        'comparison_df': comparison_df,
        'validation_report': validation_report
    }


def interactive_demo():
    """Interactive demo for user input"""
    
    print("=" * 80)
    print("INTERACTIVE FAKE NEWS DETECTION DEMO")
    print("=" * 80)
    print("\nThis demo allows you to test the system with your own text inputs.")
    print("Type 'quit' to exit the demo.")
    
    # Train a model first
    print("\nTraining model...")
    df = create_sample_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.3, random_state=42
    )
    
    detector = FakeNewsDetector(model_type='logistic_regression')
    detector.fit(X_train, y_train)
    
    # Evaluate on test set
    metrics = detector.evaluate(X_test, y_test)
    print(f"Model trained with accuracy: {metrics['accuracy']:.4f}")
    
    print("\n" + "="*50)
    print("Enter text to classify (or 'quit' to exit):")
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() == 'quit':
            print("Exiting demo...")
            break
        
        if not user_input:
            print("Please enter some text.")
            continue
        
        # Make prediction
        prediction = detector.predict([user_input])[0]
        probabilities = detector.predict_proba([user_input])[0]
        confidence = np.max(probabilities)
        
        print(f"\nPrediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        
        # Show probability distribution
        for label, prob in zip(detector.label_encoder.classes_, probabilities):
            print(f"P({label}): {prob:.4f}")
        
        # Feature importance for this prediction (if available)
        importance = detector.get_feature_importance(top_n=5)
        if importance is not None:
            print(f"\nTop contributing features:")
            for idx, row in importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")


def batch_prediction_demo():
    """Demo for batch prediction on multiple texts"""
    
    print("=" * 80)
    print("BATCH PREDICTION DEMO")
    print("=" * 80)
    
    # Sample batch of texts
    batch_texts = [
        "Scientists discover new method for renewable energy storage",
        "BREAKING: Miracle weight loss pill melts fat overnight!",
        "Government announces infrastructure spending plan",
        "SHOCKING: Celebrity death hoax confirmed - they're alive!",
        "Research team publishes findings on climate change",
        "EXCLUSIVE: Secret government documents leaked!",
        "Federal Reserve updates economic policy guidance",
        "CONSPIRACY: Moon landing was faked - new evidence!",
        "Local business expansion creates 50 new jobs",
        "ALERT: Your phone is spying on you right now!"
    ]
    
    print(f"Processing batch of {len(batch_texts)} texts...")
    
    # Train model
    df = create_sample_dataset()
    detector = FakeNewsDetector(model_type='random_forest')
    detector.fit(df['text'], df['label'])
    
    # Batch prediction
    predictions = detector.predict(batch_texts)
    probabilities = detector.predict_proba(batch_texts)
    confidences = np.max(probabilities, axis=1)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'text': batch_texts,
        'prediction': predictions,
        'confidence': confidences,
        'prob_fake': probabilities[:, 0],
        'prob_real': probabilities[:, 1]
    })
    
    print("\nBatch Prediction Results:")
    print("=" * 80)
    
    for idx, row in results_df.iterrows():
        print(f"\nText {idx + 1}:")
        print(f"Content: {row['text'][:60]}...")
        print(f"Prediction: {row['prediction']}")
        print(f"Confidence: {row['confidence']:.4f}")
        print(f"P(FAKE): {row['prob_fake']:.4f}")
        print(f"P(REAL): {row['prob_real']:.4f}")
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total texts: {len(results_df)}")
    print(f"Predicted FAKE: {len(results_df[results_df['prediction'] == 'FAKE'])}")
    print(f"Predicted REAL: {len(results_df[results_df['prediction'] == 'REAL'])}")
    print(f"Average confidence: {results_df['confidence'].mean():.4f}")
    print(f"High confidence predictions (>0.9): {len(results_df[results_df['confidence'] > 0.9])}")
    
    return results_df


if __name__ == "__main__":
    print("FAKE NEWS DETECTION SYSTEM DEMOS")
    print("=" * 80)
    print("Available demos:")
    print("1. Comprehensive Demo (full system demonstration)")
    print("2. Interactive Demo (test with your own text)")
    print("3. Batch Prediction Demo (process multiple texts)")
    
    choice = input("\nSelect demo (1-3): ").strip()
    
    if choice == "1":
        comprehensive_demo()
    elif choice == "2":
        interactive_demo()
    elif choice == "3":
        batch_prediction_demo()
    else:
        print("Invalid choice. Running comprehensive demo...")
        comprehensive_demo()
