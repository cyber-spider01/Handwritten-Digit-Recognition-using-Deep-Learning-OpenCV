import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

model.save("mnist_model.keras")

def predict_custom_image(image_path: str):
    loaded_model = tf.keras.models.load_model("mnist_model.keras")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"File not found: {image_path}")
        return
    img_resized = cv2.resize(img, (28, 28))
    if np.mean(img_resized) > 127:
        img_resized = cv2.bitwise_not(img_resized)
    img_normalized = img_resized.astype("float32") / 255.0
    input_tensor = np.expand_dims(img_normalized, axis=0)

    preds = loaded_model.predict(input_tensor)
    digit = np.argmax(preds)
    conf = np.max(preds) * 100
    print(f"Predicted: {digit} ({conf:.2f}% confidence)")

sample_indices = [0, 1, 2]
samples = X_test[sample_indices]
actual_labels = y_test[sample_indices]
predictions = model.predict(samples)

for i in range(len(sample_indices)):
    pred = np.argmax(predictions[i])
    print(f"Sample {i+1}: True={actual_labels[i]}, Predicted={pred}")