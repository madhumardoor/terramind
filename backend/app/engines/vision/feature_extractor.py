import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ResNetFeatureExtractor:
    FEATURE_DIM = 2048

    def __init__(self):
        self.device = torch.device("cpu")
        self.model = self._load_model()
        logger.info(f"ResNet50 feature extractor loaded. Output dimension: {self.FEATURE_DIM}")

    def _load_model(self):
        logger.info("Loading pre-trained ResNet50 weights...")
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        backbone = models.resnet50(weights=weights)
        feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        feature_extractor.eval()
        return feature_extractor.to(self.device)

    @torch.no_grad()
    def extract(self, tensor):
        tensor = tensor.to(self.device)
        output = self.model(tensor)
        features = output.squeeze().numpy()
        logger.debug(f"Extracted features: shape={features.shape}")
        return features

    @torch.no_grad()
    def extract_batch(self, batch_tensor):
        batch_tensor = batch_tensor.to(self.device)
        output = self.model(batch_tensor)
        features = output.squeeze(-1).squeeze(-1).numpy()
        logger.info(f"Extracted batch features: {features.shape[0]} images, {features.shape[1]} dimensions")
        return features
