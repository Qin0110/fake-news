#region REFERECES
# References of architecture building: https://www.sciencedirect.com/science/article/pii/S2667096820300070?via%3Dihub

import numpy as np
import joblib
import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

X_train, X_validate, X_test, y_train, y_validate, y_test = joblib.load('0.2_split_data.joblib')\

vectorizer = keras.layers.TextVectorization(
    max_tokens=5000,
    output_mode="int",
    output_sequence_length=500
)
vectorizer.adapt(X_train)

model = keras.models.Sequential([
    keras.layers.Input(shape=(1,), dtype='string'),
    vectorizer,
    keras.layers.Embedding(input_dim=5000, output_dim=300),
    keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.LSTM(32),
    keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_validate, y_validate)
)

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'bo-', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'bo-', label='Training Loss')
plt.plot(history.history['val_loss'], 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./model/basic_training_validation_curves.png')
plt.show()