import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
import joblib

def load_data(filepath):
    try:
        return joblib.load(filepath)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

def vectorize_data(X_train, X_test):
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    return X_train_vec, X_test_vec, tfidf

def perform_cross_validation(model, X_train_vec, y_train, cv=5):
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=cv, scoring='accuracy')
    print("Cross-validation scores for each fold:", cv_scores)
    print(f"Average cross-validation accuracy: {cv_scores.mean():.4f}")
    return cv_scores

def train_model_with_early_stopping(model, X_train_vec, y_train, X_test_vec, y_test, epochs=50, patience=5):
    
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    classes = np.unique(y_train)
    
    best_test_loss = float('inf')
    no_improve_count = 0
    best_model_weights = None

    for epoch in range(epochs):
        model.partial_fit(X_train_vec, y_train, classes=classes)

        current_train_loss = model.loss_
        current_train_acc = accuracy_score(y_train, model.predict(X_train_vec))

        test_pred_proba = model.predict_proba(X_test_vec)
        current_test_loss = log_loss(y_test, test_pred_proba)
        current_test_acc = accuracy_score(y_test, model.predict(X_test_vec))

        train_loss.append(current_train_loss)
        train_acc.append(current_train_acc)
        test_loss.append(current_test_loss)
        test_acc.append(current_test_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Test Acc: {current_test_acc:.4f} - Test Loss: {current_test_loss:.4f}")

        if current_test_loss < best_test_loss:
            best_test_loss = current_test_loss
            no_improve_count = 0
            best_model_weights = (model.coefs_, model.intercepts_)
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best Test Loss: {best_test_loss:.4f}")
                model.coefs_, model.intercepts_ = best_model_weights
                break
                
    history = {
        'train_loss': train_loss, 'test_loss': test_loss,
        'train_acc': train_acc, 'test_acc': test_acc
    }
    return model, history

def plot_learning_curves(history, save_path='MLP_Default/mlp_test_curve.png'):
    epochs_range = range(1, len(history['train_acc']) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['test_acc'], label='Test Accuracy', linestyle='--', color='orange')
    plt.title('Model Accuracy (Train vs Test)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['test_loss'], label='Test Loss', linestyle='--', color='orange')
    plt.title('Model Loss (Train vs Test)')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nGraph saved as {save_path}")
    plt.show()

def evaluate_performance(model, X_test_vec, y_test, cm_save_path='MLP_Default/mlp_confusion_matrix.png'):
    print("\nFinal Evaluation on Test Set")
    print("----------------------------")
    y_pred = model.predict(X_test_vec)
    
    print("MLP Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - MLP Classifier")
    plt.grid(False)
    plt.savefig(cm_save_path)
    print(f"Confusion matrix saved as {cm_save_path}")
    plt.show()

def main():
    data_path = "../0.2_split_data.joblib"
    data = load_data(data_path)
    if data is None: return
    
    X_train, X_validate, X_test, y_train, y_validate, y_test = data
    X_train_vec, X_test_vec, tfidf = vectorize_data(X_train, X_test)
    mlp = MLPClassifier(random_state=42)
    perform_cross_validation(mlp, X_train_vec, y_train, cv=5)
    trained_model, history = train_model_with_early_stopping(
        mlp, X_train_vec, y_train, X_test_vec, y_test, epochs=50, patience=5
    )
    plot_learning_curves(history)
    evaluate_performance(trained_model, X_test_vec, y_test)

if __name__ == "__main__":
    main()