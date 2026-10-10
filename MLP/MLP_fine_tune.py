"""Grid-searched MLP baseline on TF-IDF features.

Run from inside the MLP/ directory:  python MLP_fine_tune.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

OUT_DIR = "MLP_Fine_Tuned"
DATA_PATH = os.path.join("..", "0.2_split_data.joblib")


def load_and_vectorize_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found. Rebuild the splits with "
              "`python preprocessing.py` from the project root.")
        return None

    try:
        X_train, X_validate, X_test, y_train, y_validate, y_test = joblib.load(filepath)
    except Exception as exc:
        print(f"Error: could not read {filepath} - {exc}")
        return None

    print(f"Data loaded. Training samples: {len(X_train)}, "
          f"validation: {len(X_validate)}, test: {len(X_test)}")

    print("--- Vectorising Data ---")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec = tfidf.transform(X_validate)
    X_test_vec = tfidf.transform(X_test)
    print(f"TF-IDF matrix: {X_train_vec.shape}")

    return {
        "X_train_raw": X_train,
        "X_train_vec": X_train_vec,
        "X_val_vec": X_val_vec,
        "X_test_vec": X_test_vec,
        "y_train": y_train,
        "y_validate": y_validate,
        "y_test": y_test,
        "vectorizer": tfidf,
    }


def perform_grid_search(X_train_raw, y_train):
    print("\nStarting Grid Search")
    print("--------------------")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('mlp', MLPClassifier(max_iter=200, early_stopping=True, random_state=42))
    ])

    shared = {
        'mlp__hidden_layer_sizes': [(32,), (64,), (128,)],
        'mlp__alpha': [0.0001, 0.001, 0.01],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__learning_rate_init': [0.01, 0.001, 0.0001],
    }

    # learning_rate only affects solver='sgd'; sweeping it for Adam would
    # triple the fits for identical models.
    param_grid = [
        {**shared, 'mlp__solver': ['adam']},
        {**shared, 'mlp__solver': ['sgd'],
         'mlp__learning_rate': ['adaptive', 'constant', 'invscaling']},
    ]

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    gs.fit(X_train_raw, y_train)
    print(f"Best CV accuracy: {gs.best_score_:.4f}")
    print(f"Best Params: {gs.best_params_}")

    return {k.replace('mlp__', ''): v
            for k, v in gs.best_params_.items() if k.startswith('mlp__')}


def train_manual_mlp(X_train_vec, y_train, X_val_vec, y_validate, best_params,
                     epochs=50, patience=5):
    """Fit epoch-by-epoch, early-stopping on the *validation* split.

    Early stopping on the test split would leak the test set into model
    selection and inflate the final numbers.
    """
    print("\nStarting Manual Training Loop")
    print("----------------------------")

    mlp = MLPClassifier(
        **best_params,
        warm_start=True,
        random_state=42,
        early_stopping=False
    )

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    classes = np.unique(y_train)
    best_val_loss = float('inf')
    no_improve_count = 0
    best_model_weights = None

    for epoch in range(epochs):
        mlp.partial_fit(X_train_vec, y_train, classes=classes)

        train_loss = mlp.loss_
        train_acc = accuracy_score(y_train, mlp.predict(X_train_vec))

        val_probs = mlp.predict_proba(X_val_vec)
        val_loss = log_loss(y_validate, val_probs, labels=classes)
        val_acc = accuracy_score(y_validate, mlp.predict(X_val_vec))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Val Acc: {val_acc:.4f} "
                  f"- Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            # Copy: partial_fit mutates coefs_ in place, so storing by
            # reference would make the restore below a no-op.
            best_model_weights = ([w.copy() for w in mlp.coefs_],
                                  [b.copy() for b in mlp.intercepts_])
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"Best Validation Loss: {best_val_loss:.4f}")
                break

    if best_model_weights is not None:
        mlp.coefs_, mlp.intercepts_ = best_model_weights

    return mlp, history


def plot_learning_curves(history, save_path=os.path.join(OUT_DIR, 'mlp_test_curve.png')):
    epochs_range = range(1, len(history['train_acc']) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy',
             linestyle='--')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss',
             linestyle='--')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nGraph saved to {save_path}")
    plt.show()


def evaluate_and_save(model, vectorizer, X_test_vec, y_test):
    print("\nFinal Evaluation on Test Set")
    print("----------------------------")

    y_pred = model.predict(X_test_vec)
    y_pred_prob = model.predict_proba(X_test_vec)[:, 1]

    print("MLP Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision Score  :", precision_score(y_test, y_pred))
    print("Recall Score     :", recall_score(y_test, y_pred))
    print("F1 Score         :", f1_score(y_test, y_pred))
    print("ROC AUC          :", roc_auc_score(y_test, y_pred_prob))
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Real', 'Fake'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - MLP Classifier")
    plt.grid(False)
    plt.tight_layout()
    cm_path = os.path.join(OUT_DIR, "mlp_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.show()

    joblib.dump(vectorizer, os.path.join(OUT_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(model, os.path.join(OUT_DIR, "mlp_fake_news_model.pkl"))
    print("Model and vectorizer saved.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_and_vectorize_data(DATA_PATH)
    if data is None:
        return

    best_params = perform_grid_search(data["X_train_raw"], data["y_train"])

    model, history = train_manual_mlp(
        data["X_train_vec"], data["y_train"],
        data["X_val_vec"], data["y_validate"],
        best_params,
    )

    plot_learning_curves(history)
    evaluate_and_save(model, data["vectorizer"], data["X_test_vec"], data["y_test"])


if __name__ == "__main__":
    main()
