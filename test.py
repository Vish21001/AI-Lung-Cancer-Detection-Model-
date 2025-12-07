from tensorflow.keras.models import load_model
from preprocess import load_images
from sklearn.metrics import accuracy_score, classification_report

model = load_model("lung_cancer_lidc_model.h5")
X, y = load_images()

y_pred = (model.predict(X) > 0.5).astype("int32")
print("Accuracy:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred, target_names=["Healthy", "Cancer"]))
