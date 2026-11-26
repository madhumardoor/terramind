"""
TerraMind AI — PCA Anomaly Detector
--------------------------------------
Detects visual anomalies using PCA reconstruction error
on ResNet50 deep features.

How it works
------------
1. TRAINING (fit):
   - Collect feature vectors from many NORMAL images
   - Fit PCA on these normal features
   - PCA learns the principal components of "normal" appearance
   - Store the fitted PCA and a threshold

2. INFERENCE (predict):
   - Extract features from a new image
   - Project features into PCA space → reconstruct back
   - Measure reconstruction error (how well PCA can represent it)
   - If error > threshold → ANOMALY
   - If error <= threshold → NORMAL

Why PCA works here:
   PCA fitted on normal images captures the variance of normal appearance.
   Defective images have feature patterns not seen during fitting,
   so they cannot be well-reconstructed → high error.

Threshold selection:
   We use the 95th percentile of normal image reconstruction errors.
   This means ~5% of normal images are allowed to be false positives.
   This is adjustable based on the cost of false positives vs missed defects.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class PCAAnomalyDetector:
    """
    PCA-based anomaly detector for visual inspection.

    Requires feature vectors extracted by ResNetFeatureExtractor.

    Usage
    -----
        detector = PCAAnomalyDetector(n_components=50)
        detector.fit(normal_features)           # Train on normal images
        result = detector.predict(new_features) # Predict on new image
        detector.save(path)                     # Save for reuse
        detector.load(path)                     # Load previously fitted model
    """

    def __init__(
        self,
        n_components: int = 50,
        threshold_percentile: float = 95.0,
    ):
        """
        Args:
            n_components:          Number of PCA components to keep.
                                   50 captures most normal variance while
                                   being sensitive to subtle defects.
            threshold_percentile:  Percentile of normal errors used as
                                   the anomaly threshold. 95 = 5% false positive rate.
        """
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)

        self.threshold: Optional[float] = None
        self.is_fitted: bool = False
        self.training_errors: Optional[np.ndarray] = None

    def fit(self, normal_features: np.ndarray) -> dict:
        """
        Fit the PCA model on normal image features.

        Args:
            normal_features: Array of shape (N, 2048) where N is the
                             number of normal (defect-free) training images.
                             Minimum recommended: 20 images.

        Returns:
            Dict with training statistics:
                n_images, n_components, explained_variance_ratio,
                threshold, mean_error, std_error

        Raises:
            ValueError: If fewer than 10 training images are provided.
        """
        n_images = normal_features.shape[0]

        if n_images < 10:
            raise ValueError(
                f"Need at least 10 normal images for fitting. "
                f"Got {n_images}. Collect more normal samples."
            )

        logger.info(
            f"Fitting PCA anomaly detector on {n_images} normal images, "
            f"n_components={self.n_components}"
        )

        # Step 1: Standardise features (zero mean, unit variance)
        # This ensures no single feature dimension dominates PCA
        scaled_features = self.scaler.fit_transform(normal_features)

        # Step 2: Fit PCA on the scaled normal features
        self.pca.fit(scaled_features)

        # Step 3: Compute reconstruction errors for all training images
        # This tells us what "normal" reconstruction error looks like
        self.training_errors = self._compute_reconstruction_errors(
            normal_features
        )

        # Step 4: Set the threshold at the given percentile of normal errors
        self.threshold = float(
            np.percentile(self.training_errors, self.threshold_percentile)
        )

        self.is_fitted = True

        explained = float(self.pca.explained_variance_ratio_.sum())
        stats = {
            "n_images":               n_images,
            "n_components":           self.n_components,
            "explained_variance_ratio": round(explained, 4),
            "threshold":              round(self.threshold, 6),
            "mean_error":             round(float(self.training_errors.mean()), 6),
            "std_error":              round(float(self.training_errors.std()), 6),
        }

        logger.info(
            f"PCA fitted: {explained:.1%} variance explained, "
            f"threshold={self.threshold:.6f}"
        )
        return stats

    def _compute_reconstruction_errors(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Compute PCA reconstruction error for each feature vector.

        Process:
            1. Scale features with fitted scaler
            2. Project into PCA space (encode)
            3. Reconstruct back to original space (decode)
            4. Compute mean squared error between original and reconstructed

        Args:
            features: Array of shape (N, 2048).

        Returns:
            Array of shape (N,) with reconstruction error per image.
        """
        scaled = self.scaler.transform(features)

        # Encode: project to PCA space (N, 2048) → (N, n_components)
        encoded = self.pca.transform(scaled)

        # Decode: reconstruct back (N, n_components) → (N, 2048)
        reconstructed = self.pca.inverse_transform(encoded)

        # Mean squared error per image
        errors = np.mean((scaled - reconstructed) ** 2, axis=1)
        return errors

    def predict(self, features: np.ndarray) -> dict:
        """
        Predict whether an image is normal or anomalous.

        Args:
            features: 1D feature vector of shape (2048,) for one image,
                      or 2D array of shape (N, 2048) for a batch.

        Returns:
            For a single image:
                {
                    "is_anomaly": bool,
                    "anomaly_score": float,   # reconstruction error
                    "threshold": float,
                    "confidence": float,      # 0.0 to 1.0
                    "label": "ANOMALY" or "NORMAL"
                }

        Raises:
            RuntimeError: If the detector has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Detector is not fitted. Call fit() with normal images first, "
                "or load a previously saved model with load()."
            )

        # Handle single feature vector
        if features.ndim == 1:
            features = features.reshape(1, -1)

        errors = self._compute_reconstruction_errors(features)
        error = float(errors[0])

        is_anomaly = error > self.threshold

        # Confidence: how far the error is from the threshold
        # Normalised to 0-1 range using training error statistics
        if self.training_errors is not None:
            error_std = float(self.training_errors.std())
            if error_std > 0:
                z_score = abs(error - self.threshold) / error_std
                confidence = float(min(z_score / 3.0, 1.0))
            else:
                confidence = 1.0 if is_anomaly else 0.0
        else:
            confidence = 0.5

        result = {
            "is_anomaly":    is_anomaly,
            "anomaly_score": round(error, 6),
            "threshold":     round(self.threshold, 6),
            "confidence":    round(confidence, 4),
            "label":         "ANOMALY" if is_anomaly else "NORMAL",
        }

        logger.info(
            f"Prediction: {result['label']} "
            f"(score={result['anomaly_score']:.6f}, "
            f"threshold={result['threshold']:.6f}, "
            f"confidence={result['confidence']:.2%})"
        )
        return result

    def save(self, save_dir: Path) -> dict:
        """
        Save the fitted model to disk.

        Saves three files:
            model.pkl   — PCA and scaler objects
            metadata.json — training statistics and configuration

        Args:
            save_dir: Directory to save model files into.

        Returns:
            Dict with paths to saved files.

        Raises:
            RuntimeError: If the detector has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted detector. Call fit() first."
            )

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save PCA and scaler as pickle
        model_path = save_dir / "pca_anomaly_detector.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "pca":       self.pca,
                "scaler":    self.scaler,
                "threshold": self.threshold,
                "training_errors": self.training_errors,
                "n_components": self.n_components,
                "threshold_percentile": self.threshold_percentile,
            }, f)

        # Save human-readable metadata
        metadata_path = save_dir / "pca_anomaly_detector_metadata.json"
        metadata = {
            "n_components":           self.n_components,
            "threshold_percentile":   self.threshold_percentile,
            "threshold":              round(self.threshold, 6),
            "mean_training_error":    round(float(self.training_errors.mean()), 6),
            "std_training_error":     round(float(self.training_errors.std()), 6),
            "n_training_images":      len(self.training_errors),
            "explained_variance":     round(
                float(self.pca.explained_variance_ratio_.sum()), 4
            ),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to: {save_dir}")
        return {
            "model_path":    str(model_path),
            "metadata_path": str(metadata_path),
        }

    def load(self, save_dir: Path) -> dict:
        """
        Load a previously fitted model from disk.

        Args:
            save_dir: Directory containing the saved model files.

        Returns:
            The metadata dict from the saved model.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        save_dir = Path(save_dir)
        model_path = save_dir / "pca_anomaly_detector.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No saved model found at: {model_path}. "
                f"Train and save a model first."
            )

        with open(model_path, "rb") as f:
            saved = pickle.load(f)

        self.pca                  = saved["pca"]
        self.scaler               = saved["scaler"]
        self.threshold            = saved["threshold"]
        self.training_errors      = saved["training_errors"]
        self.n_components         = saved["n_components"]
        self.threshold_percentile = saved["threshold_percentile"]
        self.is_fitted            = True

        metadata_path = save_dir / "pca_anomaly_detector_metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

        logger.info(
            f"Model loaded from: {save_dir} "
            f"(threshold={self.threshold:.6f})"
        )
        return metadata