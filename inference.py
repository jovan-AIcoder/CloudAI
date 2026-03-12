import numpy as np
import json
from PIL import Image
import tensorflow as tf
import os
import io

def resize(image_path, size=(256, 256)):
    img = Image.open(image_path).convert("RGB")
    img.thumbnail(size)
    new_img = Image.new("RGB", size, (255, 255, 255))
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    new_img.paste(img, (x, y))
    buffer = io.BytesIO()
    new_img.save(buffer, format="PNG")

    return buffer.getvalue()

def check_image(image_path):
    allowed_extensions = (".png", ".jpg", ".jpeg")

    ext = os.path.splitext(image_path)[1].lower()

    if ext in allowed_extensions:
        return True
    else:
        return False


class LoadModel:
    def __init__(self,
                 model_path="cloud_classifier_model.h5",
                 labels_path="class_labels.json",
                 input_size=(256, 256)):
        
        self.input_size = input_size
        
        # Load model 
        self.model = tf.keras.models.load_model(model_path, compile=False)
        
        # Load class labels 
        with open(labels_path, "r") as f:
            class_labels = json.load(f)
        
        if isinstance(class_labels, dict):
            self.class_names = [class_labels[str(i)] for i in range(len(class_labels))]
        else:
            self.class_names = class_labels


    def _preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.input_size)
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array


    def inference(self, image_path):
        img_array = self._preprocess(image_path)
        
        predictions = self.model.predict(img_array, verbose=0)
        probabilities = predictions[0]
        
        top_index = np.argmax(probabilities)
        class_name = self.class_names[top_index]
        confidence = float(probabilities[top_index])
        
        return class_name, confidence