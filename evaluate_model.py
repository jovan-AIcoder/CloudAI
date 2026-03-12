from tensorflow import keras
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sklearn.metrics as met
from sklearn.preprocessing import label_binarize
model = keras.models.load_model("cloud_classifier_model.h5")
model.summary()
with open("class_labels.json","r") as f:
    class_labels = json.load(f)

class_names = []
for i in range(len(class_labels)):
    class_names.append(class_labels[str(i)])

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    "dataset/clouds_test",
    target_size=(256,256),
    batch_size=32,
    class_mode="sparse",
    shuffle=False
)
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob,axis=1)
cm = met.confusion_matrix(y_true,y_pred)
print("Confusion Matrix: \n",cm)
print("Classification report: \n")
print(met.classification_report(y_true,y_pred,target_names=class_names))
y_true_bin = label_binarize(y_true,classes=range(len(class_names)))
fpr = {}
tpr = {}
roc_auc = {}
for i in range(len(class_names)):
    fpr[i],tpr[i],_ = met.roc_curve(y_true_bin[:,i],y_pred_prob[:,i])
    roc_auc[i] = met.auc(fpr[i],tpr[i])
plt.figure(figsize=(8,6))
for i in range(len(class_names)):
    plt.plot(fpr[i],tpr[i],label=f"{class_names[i]}(AUC = {roc_auc[i]:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Cloud Classification")
plt.legend()
plt.savefig('roc_curve.png')
plt.show()