"""Default-hyperparameter MLP baseline on TF-IDF features.

Run from inside the MLP/ directory:  python MLP_Default.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             classification_report, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

OUT_DIR = "MLP_Default"
DATA_PATH = os.path.join("..", "0.2_split_data.joblib")


def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found. Rebuild the splits with "
              "`python preprocessing.py` from the project root.")
        return None
    try:
        return joblib.load(filepath)
    except Exception as exc:
        print(f"Error: could not read {filepath} - {exc}")
        return None


def vectorize_data(X_train, X_validate, X_test):
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec = tfidf.transform(X_validate)
    X_test_vec = tfidf.transform(X_test)
    return X_train_vec, X_val_vec, X_test_vec, tfidf


def perform_cross_validation(model, X_train_vec, y_train, cv=5):
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=cv,
                                scoring='accuracy')
    print("Cross-validation scores for each fold:", cv_scores)
    print(f"Average cross-validation accuracy: {cv_scores.mean():.4f}")
    return cv_scores


def train_model_with_early_stopping(model, X_train_vec, y_train, X_val_vec,
                                    y_validate, epochs=50, patience=5):
    """Fit epoch-by-epoch, early-stopping on the *validation* split.

    Using the test split here would leak it into model selection.
    """
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    classes = np.unique(y_train)

    best_val_loss = float('inf')
    no_improve_count = 0
    best_model_weights = None

    for epoch in range(epochs):
        model.partial_fit(X_train_vec, y_train, classes=classes)

        current_train_loss = model.loss_
        current_train_acc = accuracy_score(y_train, model.predict(X_train_vec))

        val_pred_proba = model.predict_proba(X_val_vec)
        current_val_loss = log_loss(y_validate, val_pred_proba, labels=classes)
        current_val_acc = accuracy_score(y_validate, model.predict(X_val_vec))

        train_loss.append(current_train_loss)
        train_acc.append(current_train_acc)
        val_loss.append(current_val_loss)
        val_acc.append(current_val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Val Acc: {current_val_acc:.4f} "
                  f"- Val Loss: {current_val_loss:.4f}")

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            no_improve_count = 0
            # Copy: partial_fit mutates coefs_ in place, so storing by
            # reference would make the restore below a no-op.
            best_model_weights = ([w.copy() for w in model.coefs_],
                                  [b.copy() for b in model.intercepts_])
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"Best Validation Loss: {best_val_loss:.4f}")
                break

    if best_model_weights is not None:
        model.coefs_, model.intercepts_ = best_model_weights

    history = {
        'train_loss': train_loss, 'val_loss': val_loss,
        'train_acc': train_acc, 'val_acc': val_acc
    }
    return model, history


def plot_learning_curves(history, save_path=os.path.join(OUT_DIR, 'mlp_test_curve.png')):
    epochs_range = range(1, len(history['train_acc']) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy',
             linestyle='--', color='orange')
    plt.title('Model Accuracy (Train vs Validation)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss',
             linestyle='--', color='orange')
    plt.title('Model Loss (Train vs Validation)')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nGraph saved as {save_path}")
    plt.show()


def evaluate_performance(model, X_test_vec, y_test,
                         cm_save_path=os.path.join(OUT_DIR, 'mlp_confusion_matrix.png'),
                         roc_save_path=os.path.join(OUT_DIR, 'mlp_roc_curve.png')):
    print("\nFinal Evaluation on Test Set")
    print("----------------------------")
    y_pred = model.predict(X_test_vec)
    y_pred_prob = model.predict_proba(X_test_vec)[:, 1]

    print("MLP Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision Score  :", precision_score(y_test, y_pred))
    print("Recall Score     :", recall_score(y_test, y_pred))
    print("F1 Score         :", f1_score(y_test, y_pred))
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
    plt.savefig(cm_save_path)
    print(f"Confusion matrix saved as {cm_save_path}")
    plt.show()

    # ROC needs continuous scores, not thresholded labels.
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

    ax.plot(fpr, tpr, color='#2ca02c', lw=3,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
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
    plt.savefig(roc_save_path)
    print(f"ROC curve saved as {roc_save_path}")
    plt.show()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_data(DATA_PATH)
    if data is None:
        return

    X_train, X_validate, X_test, y_train, y_validate, y_test = data
    X_train_vec, X_val_vec, X_test_vec, tfidf = vectorize_data(
        X_train, X_validate, X_test
    )

    mlp = MLPClassifier(random_state=42)
    perform_cross_validation(mlp, X_train_vec, y_train, cv=5)

    trained_model, history = train_model_with_early_stopping(
        mlp, X_train_vec, y_train, X_val_vec, y_validate, epochs=50, patience=5
    )

    plot_learning_curves(history)
    evaluate_performance(trained_model, X_test_vec, y_test)


if __name__ == "__main__":
    main()
