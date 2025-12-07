from preprocess import load_images
from model import create_model
from sklearn.model_selection import train_test_split

# Load LIDC dataset
X, y = load_images()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = create_model()
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val))

# Save model
model.save("lung_cancer_lidc_model.h5")
