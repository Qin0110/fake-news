"""Evaluate a saved model against a held-out test split.

Examples
--------
    python test.py
    python test.py --model model/best_cnn_rnn_model.keras --data 0.3_split_data.joblib
    python test.py --model MLP/MLP_Fine_Tuned/mlp_fake_news_model.pkl \
                   --vectorizer MLP/MLP_Fine_Tuned/tfidf_vectorizer.pkl
"""

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='model/best_cnn_rnn_model.keras',
                        help='Path to a .keras model or a pickled scikit-learn model.')
    parser.add_argument('--data', default='0.2_split_data.joblib',
                        help='Split file produced by preprocessing.py.')
    parser.add_argument('--vectorizer', default=None,
                        help='TF-IDF vectorizer .pkl, required for scikit-learn models.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability at or above which a document is called fake.')
    parser.add_argument('--roc', default=None,
                        help='Optional path to save the ROC curve as a PNG.')
    return parser.parse_args()


def load_scores(args):
    """Return ``(y_test, y_pred_prob)`` for whichever model type was given."""
    for path in (args.data, args.model):
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' not found.")

    X_train, X_validate, X_test, y_train, y_validate, y_test = joblib.load(args.data)
    print(f"Loaded {args.data}: test set has {len(X_test)} documents.")

    if args.model.endswith('.keras'):
        import keras
        model = keras.models.load_model(args.model)
        model.summary()
        y_pred_prob = model.predict(X_test).reshape(-1)
    else:
        if not args.vectorizer:
            raise ValueError(
                "--vectorizer is required when evaluating a pickled "
                "scikit-learn model (the text must be TF-IDF transformed first)."
            )
        model = joblib.load(args.model)
        vectorizer = joblib.load(args.vectorizer)
        X_test_vec = vectorizer.transform(X_test)
        y_pred_prob = model.predict_proba(X_test_vec)[:, 1]

    return np.asarray(y_test), np.asarray(y_pred_prob, dtype=float)


def plot_roc(fpr, tpr, roc_auc, save_path=None):
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

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    plt.show()


def main():
    args = parse_args()
    y_test, y_pred_prob = load_scores(args)
    y_pred = (y_pred_prob >= args.threshold).astype(int)

    print(f"\n--- Final Evaluation On Test Set (threshold={args.threshold}) ---")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 score :", f1_score(y_test, y_pred))
    # Probabilities, not hard labels: thresholded input collapses the curve.
    print("ROC AUC  :", roc_auc_score(y_test, y_pred_prob))
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    print("Confusion Matrix (rows = actual, cols = predicted):\n",
          confusion_matrix(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plot_roc(fpr, tpr, auc(fpr, tpr), args.roc)


if __name__ == "__main__":
    main()
