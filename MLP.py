import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

X_train, X_validate, X_test, y_train, y_validate, y_test = joblib.load('0.3_split_data.joblib')


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=2000)),
    ('mlp', MLPClassifier(max_iter=50, early_stopping=True, random_state=42))
])

param_grid = {
    'mlp__hidden_layer_sizes': [(32,), (64,), (32, 32)],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__solver': ['adam'],
    'mlp__learning_rate': ['adaptive', 'constant'],
    'mlp__learning_rate_init': [0.0001]
}

gs = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=3
)

gs.fit(X_train, y_train)

print(f"\nBest Params: {gs.best_params_}")

best_pipe = gs.best_estimator_
y_val_pred = best_pipe.predict(X_validate)
print(f"Validation Accuracy: {accuracy_score(y_validate, y_val_pred):.4f}")

best_params_mlp = {k.replace('mlp__', ''): v for k, v in gs.best_params_.items() if k.startswith('mlp__')}

tfidf = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_validate)

mlp = MLPClassifier(
    **best_params_mlp,
    max_iter=1,
    warm_start=True,
    random_state=42,
    early_stopping=False
)

train_loss, val_loss = [], []
train_acc, val_acc = [], []
classes = [0, 1]
epochs = 20

for epoch in range(epochs):
    mlp.partial_fit(X_train_vec, y_train, classes=classes)
    
    train_loss.append(mlp.loss_)
    val_pred_proba = mlp.predict_proba(X_val_vec)
    val_loss.append(log_loss(y_validate, val_pred_proba))
    
    train_pred = mlp.predict(X_train_vec)
    val_pred = mlp.predict(X_val_vec)
    train_acc.append(accuracy_score(y_train, train_pred))
    val_acc.append(accuracy_score(y_validate, val_pred))
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{epochs} - Val Acc: {val_acc[-1]:.4f}")


epochs_range = range(1, epochs + 1)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', linestyle='--')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss', linestyle='--')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

joblib.dump(mlp, "mlp_fake_news_model.pkl")

