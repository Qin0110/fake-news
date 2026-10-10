"""Baseline (untuned) CNN + LSTM model, for comparison against cnn+rnn.py.

Architecture reference:
https://www.sciencedirect.com/science/article/pii/S2667096820300070?via%3Dihub
"""

import os

import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

MODEL_DIR = "model"
DATA_FILE = "0.2_split_data.joblib"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"'{DATA_FILE}' not found. Rebuild the splits with "
        "`python preprocessing.py` (requires fake_news.csv)."
    )

X_train, X_validate, X_test, y_train, y_validate, y_test = joblib.load(DATA_FILE)
print(f"train={len(X_train)}, validate={len(X_validate)}, test={len(X_test)}")

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

y_pred_prob = model.predict(X_test).reshape(-1)
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 score :", f1_score(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

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
plt.savefig(os.path.join(MODEL_DIR, 'basic_training_validation_curves.png'))
plt.show()

# Probabilities, not 0/1 labels: those collapse the curve to 3 points.
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

print("Model ROC AUC Score:", roc_auc_score(y_test, y_pred_prob))
print(f"Exact AUC Score: {roc_auc}")

fig, ax = plt.subplots(figsize=(8, 6))
cmap = plt.get_cmap('Greens')
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient)).T
img = ax.imshow(gradient, extent=[0, 1, 0, 1], origin='lower', cmap=cmap,
                aspect='auto', alpha=0.8)

poly_points = list(zip(fpr, tpr)) + [(1, 0), (0, 0)]
poly = Polygon(poly_points, facecolor='none', edgecolor='none')
ax.add_patch(poly)

img.set_clip_path(poly)

ax.plot(fpr, tpr, color='#2ca02c', lw=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='#2f4f4f', lw=2, linestyle='--',
        label='Random Guessing')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (False Alarms)', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=12)
ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=14)
ax.legend(loc="lower right", fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'basic_roc_curve.png'))
plt.show()
