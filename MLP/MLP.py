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
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('mlp', MLPClassifier(max_iter=200, early_stopping=True, random_state=42))
])

param_grid = {
    'mlp__hidden_layer_sizes': [(32,), (64,), (128,)],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__solver': ['adam', 'sgd'],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__learning_rate': ['adaptive', 'constant', 'invscaling'],
    'mlp__learning_rate_init': [0.01, 0.001, 0.0001]
}

gs = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

gs.fit(X_train, y_train)

print(f"\nBest Params: {gs.best_params_}")
best_pipe = gs.best_estimator_


best_params_mlp = {k.replace('mlp__', ''): v for k, v in gs.best_params_.items() if k.startswith('mlp__')}

tfidf = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_validate)
X_test_vec = tfidf.transform(X_test)  

mlp = MLPClassifier(
    **best_params_mlp,
    warm_start=True,
    random_state=42,
    early_stopping=False
)

train_loss, val_loss = [], []
train_acc, val_acc = [], []
classes = [0, 1]
epochs = 50

best_val_loss = float('inf')
patience = 5
no_improve_count = 0
best_model_weights = None

for epoch in range(epochs):
    mlp.partial_fit(X_train_vec, y_train, classes=classes)

    current_train_loss = mlp.loss_
    current_train_acc = accuracy_score(y_train, mlp.predict(X_train_vec))

    val_pred_proba = mlp.predict_proba(X_val_vec)
    current_val_loss = log_loss(y_validate, val_pred_proba)
    current_val_acc = accuracy_score(y_validate, mlp.predict(X_val_vec))

    train_loss.append(current_train_loss)
    train_acc.append(current_train_acc)
    val_loss.append(current_val_loss)
    val_acc.append(current_val_acc)

    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Val Acc: {current_val_acc:.4f} - Val Loss: {current_val_loss:.4f}")

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        no_improve_count = 0
        best_model_weights = (mlp.coefs_, mlp.intercepts_)
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best Validation Loss: {best_val_loss:.4f}")

            mlp.coefs_, mlp.intercepts_ = best_model_weights
            break

epochs_range = range(1, len(train_acc) + 1)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', linestyle='--')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1.05)
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
plt.savefig('mlp_training_curve.png')
plt.show()

joblib.dump(mlp, "mlp_fake_news_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")


y_test_pred = mlp.predict(X_test_vec)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_log_loss = log_loss(y_test, mlp.predict_proba(X_test_vec))

print(f"Final Test Accuracy: {test_accuracy:.4f}")
print(f"Final Test Log Loss: {test_log_loss:.4f}")