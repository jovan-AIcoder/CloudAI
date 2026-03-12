# import library
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
# preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    "dataset/clouds_train",
    target_size=(256,256),
    batch_size=32,
    class_mode="sparse"
)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    "dataset/clouds_test",
    target_size=(256,256),
    batch_size=32,
    class_mode="sparse",
    shuffle=False 
)
num_classes = train_generator.num_classes
print("Number of cloud classes: ",num_classes)
# model
model = keras.Sequential([
    keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(256,256,3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(128,(3,3),activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(16,activation='relu'),
    keras.layers.Dense(num_classes,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# Fit
model.fit(
    train_generator,
    epochs=30,
    validation_data=test_generator
)
# saving the model and class indices
model.save('cloud_classifier_model.h5')
class_indices = train_generator.class_indices
labels = {v: k for k, v in class_indices.items()}
with open("class_labels.json","w") as f:
    json.dump(labels,f,indent=4)