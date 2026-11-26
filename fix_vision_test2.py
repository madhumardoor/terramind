content = """\
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.core.logging_config import setup_logging
setup_logging(log_level="INFO")

import numpy as np
from PIL import Image
from app.engines.vision.image_preprocessor import ImagePreprocessor
from app.engines.vision.feature_extractor import ResNetFeatureExtractor
from app.engines.vision.anomaly_detector import PCAAnomalyDetector

print("\\n" + "="*60)
print("TerraMind AI - Vision Engine Test")
print("="*60)

print("\\n[1] Creating synthetic test images...")
normal_images = []
for i in range(20):
    arr = np.random.randint(200, 255, (224, 224, 3), dtype=np.uint8)
    normal_images.append(Image.fromarray(arr))
print(f"    Created {len(normal_images)} normal images")

defect_image = Image.fromarray(
    np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
)
print("    Created 1 defect image (very dark)")

print("\\n[2] Preprocessing images...")
preprocessor = ImagePreprocessor()
normal_tensors = preprocessor.preprocess_batch(normal_images)
defect_tensor = preprocessor.preprocess(defect_image)
print(f"    Normal batch shape : {normal_tensors.shape}")
print(f"    Defect tensor shape: {defect_tensor.shape}")

print("\\n[3] Extracting ResNet50 features...")
extractor = ResNetFeatureExtractor()
normal_features = extractor.extract_batch(normal_tensors)
defect_features = extractor.extract(defect_tensor)
print(f"    Normal features shape : {normal_features.shape}")
print(f"    Defect features shape : {defect_features.shape}")

print("\\n[4] Fitting PCA anomaly detector on normal images...")
detector = PCAAnomalyDetector(n_components=10)
stats = detector.fit(normal_features)
print(f"    Variance explained : {stats['explained_variance_ratio']:.1%}")
print(f"    Threshold          : {stats['threshold']:.6f}")
print(f"    Mean training error: {stats['mean_error']:.6f}")

print("\\n[5] Running predictions...")
normal_result = detector.predict(normal_features[0])
defect_result = detector.predict(defect_features)
print(f"    Normal image -> {normal_result['label']} (score={normal_result['anomaly_score']:.6f})")
print(f"    Defect image -> {defect_result['label']} (score={defect_result['anomaly_score']:.6f})")

print("\\n[6] Saving and reloading model...")
save_dir = Path(__file__).resolve().parent.parent / "data" / "processed" / "vision_models"
save_dir.mkdir(parents=True, exist_ok=True)
detector.save(save_dir)
print(f"    Saved to: {save_dir}")

new_detector = PCAAnomalyDetector()
new_detector.load(save_dir)
reload_result = new_detector.predict(defect_features)
print(f"    Reloaded model prediction: {reload_result['label']}")

print("\\n" + "="*60)
print("Vision engine test complete.")
print("="*60)
"""

with open("scripts/test_vision_engine.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Test script written successfully")