import cv2
import numpy as np
import tensorflow as tf

from src.config import IMG_SIZE, MODEL_PATH


def preprocess_image(image_path):
    """Load image from disk, resize and normalize for model input"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def postprocess_mask(pred):
    """Convert raw model output to binary mask (0 or 255)"""
    mask = pred[0].squeeze()
    return (mask > 0.5).astype(np.uint8) * 255


if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    img = preprocess_image("assets/test.jpg")
    pred = model.predict(img)
    mask = postprocess_mask(pred)

    cv2.imwrite("assets/pred_mask.png", mask)
    print("Mask saved to assets/pred_mask.png")
