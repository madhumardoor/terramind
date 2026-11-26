"""
TerraMind AI - Vision Inspection API
--------------------------------------
Exposes the computer vision engine as REST endpoints.

Endpoints
---------
POST /vision/inspect
    Upload an image and receive an anomaly detection result.
    Returns label (NORMAL/ANOMALY), score, confidence, threshold.

POST /vision/train
    Train the anomaly detector on a set of uploaded normal images.
    Saves the fitted model to data/processed/vision_models/.

GET /vision/model/info
    Return metadata about the currently loaded model.

GET /vision/health
    Check whether a fitted model is available.
"""

import io
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging_config import get_logger
from app.engines.vision.anomaly_detector import PCAAnomalyDetector
from app.engines.vision.feature_extractor import ResNetFeatureExtractor
from app.engines.vision.image_preprocessor import ImagePreprocessor

logger = get_logger(__name__)

router = APIRouter(prefix="/vision", tags=["Vision"])

# Model storage directory
MODEL_DIR = Path("../data/processed/vision_models")

# Module-level singletons — loaded once, reused across requests
_preprocessor: Optional[ImagePreprocessor] = None
_extractor: Optional[ResNetFeatureExtractor] = None
_detector: Optional[PCAAnomalyDetector] = None


def _get_preprocessor() -> ImagePreprocessor:
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = ImagePreprocessor()
    return _preprocessor


def _get_extractor() -> ResNetFeatureExtractor:
    global _extractor
    if _extractor is None:
        logger.info("Loading ResNet50 feature extractor...")
        _extractor = ResNetFeatureExtractor()
    return _extractor


def _get_detector() -> Optional[PCAAnomalyDetector]:
    global _detector
    if _detector is None:
        model_path = MODEL_DIR / "pca_anomaly_detector.pkl"
        if model_path.exists():
            logger.info("Loading saved anomaly detector model...")
            _detector = PCAAnomalyDetector()
            _detector.load(MODEL_DIR)
        else:
            logger.warning("No fitted model found. Train the model first.")
    return _detector


def _read_upload_as_pil(upload: UploadFile) -> Image.Image:
    """
    Read an uploaded file into a PIL Image.
    Validates that it is a supported image format.
    """
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}

    if upload.content_type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {upload.content_type}. "
                   f"Allowed: jpeg, png, bmp, tiff.",
        )

    contents = upload.file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read image file: {e}",
        )


class InspectionResult(BaseModel):
    filename: str
    label: str
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    confidence: float
    image_width: int
    image_height: int
    model_loaded: bool


class TrainingResult(BaseModel):
    message: str
    n_images_used: int
    explained_variance: float
    threshold: float
    mean_error: float
    model_saved_to: str


@router.get("/health")
async def vision_health() -> dict:
    """
    Check whether a fitted anomaly detection model is available.
    """
    detector = _get_detector()
    model_path = MODEL_DIR / "pca_anomaly_detector.pkl"

    return {
        "model_available": detector is not None,
        "model_path": str(model_path),
        "model_fitted": detector.is_fitted if detector else False,
    }


@router.get("/model/info")
async def get_model_info() -> dict:
    """
    Return metadata about the currently loaded anomaly detection model.
    """
    metadata_path = MODEL_DIR / "pca_anomaly_detector_metadata.json"

    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No model metadata found. Train the model first via POST /vision/train",
        )

    import json
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "model_type": "PCA Anomaly Detector with ResNet50 Features",
        "feature_extractor": "ResNet50 (ImageNet pre-trained, 2048-dim output)",
        "metadata": metadata,
    }


@router.post("/inspect", response_model=InspectionResult)
async def inspect_image(
    file: UploadFile = File(..., description="Image file to inspect (JPEG or PNG)"),
) -> InspectionResult:
    """
    Inspect an uploaded image for anomalies.

    Upload a JPEG or PNG image. The endpoint will:
    1. Preprocess the image (resize to 224x224, normalise)
    2. Extract ResNet50 deep features (2048 dimensions)
    3. Run PCA anomaly detection
    4. Return label, score, confidence and threshold

    Requires a fitted model — run POST /vision/train first.
    """
    logger.info(f"Inspection requested for: {file.filename}")

    detector = _get_detector()
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="No fitted model available. POST to /vision/train first.",
        )

    # Read and validate the uploaded image
    pil_image = _read_upload_as_pil(file)
    width, height = pil_image.size

    # Preprocess and extract features
    preprocessor = _get_preprocessor()
    extractor = _get_extractor()

    tensor = preprocessor.preprocess(pil_image)
    features = extractor.extract(tensor)

    # Run anomaly detection
    result = detector.predict(features)

    logger.info(
        f"Inspection result for {file.filename}: "
        f"{result['label']} (confidence={result['confidence']:.1%})"
    )

    return InspectionResult(
        filename=file.filename or "uploaded_image",
        label=result["label"],
        is_anomaly=result["is_anomaly"],
        anomaly_score=result["anomaly_score"],
        threshold=result["threshold"],
        confidence=result["confidence"],
        image_width=width,
        image_height=height,
        model_loaded=True,
    )


@router.post("/train", response_model=TrainingResult)
async def train_model(
    files: list[UploadFile] = File(
        ...,
        description="10 or more normal (defect-free) images for training",
    ),
) -> TrainingResult:
    """
    Train the anomaly detector on uploaded normal images.

    Upload at least 10 defect-free images. The detector will:
    1. Extract ResNet50 features from each image
    2. Fit PCA on the combined feature matrix
    3. Set the anomaly threshold at the 95th percentile
    4. Save the model to data/processed/vision_models/

    After training, use POST /vision/inspect to test new images.
    """
    global _detector

    if len(files) < 10:
        raise HTTPException(
            status_code=422,
            detail=f"At least 10 normal images required for training. "
                   f"Got {len(files)}.",
        )

    logger.info(f"Training anomaly detector on {len(files)} images...")

    preprocessor = _get_preprocessor()
    extractor = _get_extractor()

    # Process all uploaded images
    all_features = []
    for upload in files:
        pil_image = _read_upload_as_pil(upload)
        tensor = preprocessor.preprocess(pil_image)
        features = extractor.extract(tensor)
        all_features.append(features)

    feature_matrix = np.stack(all_features, axis=0)
    logger.info(f"Feature matrix shape: {feature_matrix.shape}")

    # Fit the detector
    detector = PCAAnomalyDetector(n_components=50)
    stats = detector.fit(feature_matrix)

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    detector.save(MODEL_DIR)

    # Update singleton
    _detector = detector

    return TrainingResult(
        message=f"Model trained successfully on {len(files)} images.",
        n_images_used=len(files),
        explained_variance=stats["explained_variance_ratio"],
        threshold=stats["threshold"],
        mean_error=stats["mean_error"],
        model_saved_to=str(MODEL_DIR),
    )
