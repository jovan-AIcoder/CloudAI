import numpy as np
import json
import os
import io

from PIL import Image
import tensorflow as tf
import matplotlib.cm as cm


# ==========================================
# Convert image file to bytes
# ==========================================
def image_to_bytes(path, size=None):
    img = Image.open(path).convert("RGB")

    if size:
        img = img.resize(size)

    bio = io.BytesIO()
    img.save(bio, format="PNG")

    return bio.getvalue()


# ==========================================
# Resize image for preview
# ==========================================
def resize(image_path, size=(256, 256)):
    img = Image.open(image_path).convert("RGB")

    img.thumbnail(size)

    new_img = Image.new(
        "RGB",
        size,
        (255, 255, 255)
    )

    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2

    new_img.paste(
        img,
        (x, y)
    )

    buffer = io.BytesIO()

    new_img.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


# ==========================================
# Check image extension
# ==========================================
def check_image(image_path):
    allowed_extensions = (
        ".png",
        ".jpg",
        ".jpeg"
    )

    ext = os.path.splitext(
        image_path
    )[1].lower()

    return ext in allowed_extensions


# ==========================================
# CloudAI Model
# ==========================================
class LoadModel:

    def __init__(
            self,
            model_path="cloud_classifier_model.h5",
            labels_path="class_labels.json",
            input_size=(256, 256)
    ):

        self.input_size = input_size

        # Load CNN model
        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        # Load class labels
        with open(
                labels_path,
                "r"
        ) as f:

            class_labels = json.load(f)

        if isinstance(
                class_labels,
                dict
        ):

            self.class_names = [
                class_labels[str(i)]
                for i in range(
                    len(class_labels)
                )
            ]

        else:
            self.class_names = class_labels

    # ======================================
    # Image preprocessing
    # ======================================
    def _preprocess(self, image_path):

        img = Image.open(
            image_path
        ).convert("RGB")

        img = img.resize(
            self.input_size
        )

        img_array = np.array(
            img,
            dtype=np.float32
        ) / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        return img_array

    # ======================================
    # Cloud classification
    # ======================================
    def inference(self, image_path):

        img_array = self._preprocess(
            image_path
        )

        predictions = self.model.predict(
            img_array,
            verbose=0
        )

        probabilities = predictions[0]

        top_index = np.argmax(
            probabilities
        )

        class_name = (
            self.class_names[top_index]
        )

        confidence = float(
            probabilities[top_index]
        )

        return (
            class_name,
            confidence
        )

    # ======================================
    # Grad-CAM
    # ======================================
    def GradCAM(
        self,
        image_path,
        alpha=0.4):

        # ======================
        # Load image
        # ======================
        img = Image.open(
            image_path
        ).convert("RGB")

        img = img.resize(
            self.input_size
        )

        original = np.array(
            img,
            dtype=np.uint8
        )

        img_array = (
            original.astype(
                np.float32
            ) / 255.0
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Build model graph
        _ = self.model(img_array)

        # ======================
        # Last Conv Layer
        # ======================
        last_conv_layer = (
            self.model.get_layer(
                "conv2d_2"
            )
        )

        # Model from input to conv2d_2
        last_conv_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=last_conv_layer.output
        )

        # Classifier head
        classifier_input = tf.keras.Input(
            shape=last_conv_layer.output.shape[1:]
        )

        x = classifier_input

        for layer in self.model.layers[5:]:
            x = layer(x)

        classifier_model = tf.keras.Model(
            classifier_input,
            x
        )

        # ======================
        # Compute Gradients
        # ======================
        with tf.GradientTape() as tape:

            conv_output = last_conv_model(
                img_array
            )

            tape.watch(
                conv_output
            )

            predictions = classifier_model(
                conv_output
            )

            pred_index = tf.argmax(
                predictions[0]
            )

            class_channel = (
                predictions[
                    :,
                    pred_index
                ]
            )

        grads = tape.gradient(
            class_channel,
            conv_output
        )

        # Safety check
        if grads is None:
            raise RuntimeError(
                "Gradients are None."
            )

        # ======================
        # Grad-CAM Equation
        # α_k
        # ======================
        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        conv_output = (
            conv_output[0]
        )

        heatmap = tf.reduce_sum(
            conv_output
            * pooled_grads,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        heatmap /= (
            tf.reduce_max(
                heatmap
            ) + 1e-10
        )

        heatmap = heatmap.numpy()

        # ======================
        # Resize Heatmap
        # ======================
        heatmap = Image.fromarray(
            np.uint8(
                heatmap * 255
            )
        )

        heatmap = heatmap.resize(
            self.input_size
        )

        heatmap = (
            np.array(
                heatmap
            ) / 255.0
        )

        # ======================
        # Apply JET Colormap
        # ======================
        jet = cm.get_cmap(
            "jet"
        )

        jet_colors = jet(
            heatmap
        )[:, :, :3]

        jet_colors = (
            jet_colors * 255
        ).astype(
            np.uint8
        )

        # ======================
        # Overlay
        # ======================
        overlay = (
            alpha
            * jet_colors
            + (1-alpha)
            * original
        )

        overlay = np.clip(
            overlay,
            0,
            255
        ).astype(
            np.uint8
        )

        return overlay