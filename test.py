import joblib   
import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import json

X_train, X_validate, X_test, y_train, y_validate, y_test = joblib.load('0.2_split_data.joblib')
# model = joblib.load('mlp_fake_news_model_0.2.pkl')
# vectorizer = joblib.load('tfidf_vectorizer_0.2.pkl')
model = keras.models.load_model('./model/best_cnn_rnn_model.keras')

# X_test = vectorizer.transform(X_test)
y_pred_prob = model.predict(X_test).ravel()  # Flatten to 1D array
y_pred = (y_pred_prob > 0.5).astype(int)

print("First 10 predictions:", y_pred_prob[:10])



print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# model.summary()

# config = model.get_config()

# keras.utils.plot_model(
#     model,
#     to_file='model_architecture.png',
#     show_shapes=True,      # Shows inputs/output dimensions (Very useful!)
#     show_layer_names=True, # Shows 'conv1d', 'lstm', etc.
#     show_layer_activations=True # Shows 'relu', 'tanh', etc.
# )
# print("--- Final Evaluation On Test Set ---")
# print("MLP Model Accuracy:", accuracy_score(y_test, y_pred))
# print("MLP Model ROC AUC Score:", roc_auc_score(y_test, y_pred))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (False Alarms)')
plt.ylabel('True Positive Rate (Recall / Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) for Fake News Model')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()