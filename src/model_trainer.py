"""
Model trainer for the Intelligent Document Classifier.

This module handles fetching data via the DataPipeline, splitting it into
train/test sets, training a classification pipeline, and saving the model.
"""

import logging
import os
from typing import Tuple

import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src.data_pipeline import DataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_and_evaluate_model() -> None:
    """
    Fetch data, train a classification model, evaluate it, and save the pipeline.
    """
    logger.info("Starting model training and evaluation process.")

    try:
        # 1. Fetch and clean data
        pipeline_data = DataPipeline()
        newsgroups = pipeline_data.fetch_newsgroups_data()

        logger.info("Preprocessing documents...")
        # Preprocess all documents
        X = [pipeline_data.preprocess_text(doc) for doc in newsgroups.data]
        y = newsgroups.target

        # 2. Split data: 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Data split: {len(X_train)} training, {len(X_test)} testing samples.")

        # 3. Create a scikit-learn Pipeline
        # Using TfidfVectorizer for feature extraction and Calibrated LinearSVC for confidence scores
        model_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('clf', CalibratedClassifierCV(LinearSVC(random_state=42), cv=5))
        ])

        # 4. Train (fit) the pipeline
        logger.info("Training the model pipeline...")
        model_pipeline.fit(X_train, y_train)

        # 5. Predict and evaluate
        logger.info("Evaluating model performance...")
        y_pred = model_pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=newsgroups.target_names)
        
        print("\n--- Model Evaluation Report ---")
        print(report)

        # 6. Save the trained pipeline
        model_path = os.path.join("models", "document_classifier.joblib")
        logger.info(f"Saving trained model to {model_path}...")
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        joblib.dump(model_pipeline, model_path)
        
        logger.info("Model training and saving completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during training/evaluation: {e}")
        raise


if __name__ == '__main__':
    train_and_evaluate_model()
