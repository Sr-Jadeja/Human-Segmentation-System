import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small

from src.config import IMG_SIZE


def conv_block(x, filters, name):
    """Two Conv2D layers with BatchNorm and ReLU activation"""
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", name=name + "_conv1")(x)
    x = layers.BatchNormalization(name=name + "_bn1")(x)
    x = layers.Activation("relu", name=name + "_act1")(x)

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", name=name + "_conv2")(x)
    x = layers.BatchNormalization(name=name + "_bn2")(x)
    x = layers.Activation("relu", name=name + "_act2")(x)
    return x


def build_model():
    """U-Net style model with MobileNetV3Small encoder and skip connections"""
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))

    # Encoder: MobileNetV3Small pretrained on ImageNet
    base = MobileNetV3Small(input_tensor=inputs, include_top=False, weights="imagenet")

    # Named skip connections — stable across TF versions
    e1 = base.get_layer("re_lu").output           # ~128x128
    e2 = base.get_layer("re_lu_3").output         # ~64x64
    e3 = base.get_layer("re_lu_6").output         # ~32x32
    e4 = base.get_layer("re_lu_12").output        # ~16x16
    e5 = base.get_layer("activation_17").output   # ~8x8

    # Decoder: upsample to skip connection size, concatenate, then conv block
    def upsample_concat(x, skip, filters, name):
        x = layers.Resizing(skip.shape[1], skip.shape[2], interpolation="bilinear")(x)
        x = layers.Concatenate()([skip, x])
        return conv_block(x, filters, name)

    x = conv_block(e5, 256, "d5")
    x = upsample_concat(x, e4, 256, "d4")
    x = upsample_concat(x, e3, 128, "d3")
    x = upsample_concat(x, e2, 64,  "d2")
    x = upsample_concat(x, e1, 32,  "d1")

    # Final resize to original input size + sigmoid binary output
    x = layers.Resizing(IMG_SIZE, IMG_SIZE, interpolation="bilinear")(x)
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    return models.Model(inputs, outputs, name="UNet_MobileNetV3")


if __name__ == "__main__":
    model = build_model()
    model.summary()
