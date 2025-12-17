import joblib   
import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

X_train, X_validate, X_test, y_train, y_validate, y_test = joblib.load('0.2_split_data.joblib')
# model = joblib.load('mlp_fake_news_model_0.2.pkl')
# vectorizer = joblib.load('tfidf_vectorizer_0.2.pkl')
# X_test = vectorizer.transform(X_test)
model = keras.models.load_model('./model/best_cnn_rnn_model.keras')
model.summary()

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("f1_score:", f1_score(y_test, y_pred))
print("recall_score:", recall_score(y_test, y_pred))
print("precision_score:", precision_score(y_test, y_pred))

# print("First 10 predictions:", y_pred_prob[:10])



# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# roc_auc = auc(fpr, tpr)

# print("--- Final Evaluation On Test Set ---")
# print("Model Accuracy:", accuracy_score(y_test, y_pred))
# print("Model ROC AUC Score:", roc_auc_score(y_test, y_pred))
# print(f"Exact AUC Score: {roc_auc}")

# fig, ax = plt.subplots(figsize=(8, 6))
# cmap = plt.get_cmap('Greens')
# gradient = np.linspace(0, 1, 256)
# gradient = np.vstack((gradient, gradient)).T
# img = ax.imshow(gradient, extent=[0, 1, 0, 1], origin='lower', cmap=cmap, aspect='auto', alpha=0.8)

# poly_points = list(zip(fpr, tpr)) + [(1, 0), (0, 0)]
# poly = Polygon(poly_points, facecolor='none', edgecolor='none')
# ax.add_patch(poly)

# img.set_clip_path(poly)

# ax.plot(fpr, tpr, color='#2ca02c', lw=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
# ax.plot([0, 1], [0, 1], color='#2f4f4f', lw=2, linestyle='--', label='Random Guessing')

# # 4. Final Touches
# ax.set_xlim([0.0, 1.0])
# ax.set_ylim([0.0, 1.05])
# ax.set_xlabel('False Positive Rate (False Alarms)', fontsize=12)
# ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=12)
# ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=14)
# ax.legend(loc="lower right", fontsize=12)
# ax.grid(alpha=0.3)

# plt.show()