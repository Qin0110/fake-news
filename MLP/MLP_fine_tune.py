import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def load_and_vectorize_data(filepath):
    print("--- Loading Data ---")
    try:
        data = joblib.load(filepath)
        X_train, X_validate, X_test, y_train, y_validate, y_test = data
        print(f"Data loaded. Training samples: {len(X_train)}")
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

    print("--- Vectorising Data ---")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec = tfidf.transform(X_validate)
    X_test_vec = tfidf.transform(X_test)
    
    return (X_train, y_train), (X_train_vec, X_val_vec, X_test_vec, y_validate, y_test), tfidf

def perform_grid_search(X_train_raw, y_train):
    print("\nStarting Grid Search")
    print("--------------------")

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
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    gs.fit(X_train_raw, y_train)
    print(f"Best Params: {gs.best_params_}")
    
    best_params = {k.replace('mlp__', ''): v for k, v in gs.best_params_.items() if k.startswith('mlp__')}
    
    return best_params

def train_manual_mlp(X_train_vec, y_train, X_test_vec, y_test, best_params):
    print("\nStarting Manual Training Loop")
    print("----------------------------")
    
    mlp = MLPClassifier(
        **best_params,
        warm_start=True,
        random_state=42,
        early_stopping=False 
    )

    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    classes = np.unique(y_train)
    epochs = 50
    patience = 5
    best_test_loss = float('inf')
    no_improve_count = 0
    best_model_weights = None

    for epoch in range(epochs):
        mlp.partial_fit(X_train_vec, y_train, classes=classes)

        train_loss = mlp.loss_
        train_acc = accuracy_score(y_train, mlp.predict(X_train_vec))
        
        test_probs = mlp.predict_proba(X_test_vec)
        test_loss = log_loss(y_test, test_probs)
        test_acc = accuracy_score(y_test, mlp.predict(X_test_vec))

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Test Acc: {test_acc:.4f} - Test Loss: {test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            no_improve_count = 0
            best_model_weights = (mlp.coefs_, mlp.intercepts_)
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best Test Loss: {best_test_loss:.4f}")
                mlp.coefs_, mlp.intercepts_ = best_model_weights
                break
                
    return mlp, history

def plot_learning_curves(history, save_path='MLP_Fine_Tuned/mlp_training_curve.png'):
    epochs_range = range(1, len(history['train_acc']) + 1)
    
    plt.figure(figsize=(14, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['test_acc'], label='Test Accuracy', linestyle='--')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['test_loss'], label='Test Loss', linestyle='--')
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
    print("\nFinal Evaluation")
    print("----------------")
    y_test_pred = model.predict(X_test_vec)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_log_loss = log_loss(y_test, model.predict_proba(X_test_vec))

    y_pred = model.predict(X_test_vec)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - MLP Classifier")
    plt.grid(False)
    plt.savefig("MLP_Fine_Tuned/mlp_confusion_matrix.png")
    plt.show()

    print("\nFinal Evaluation on Test Set")
    print("----------------------------")
    y_pred = model.predict(X_test_vec)
    print("MLP Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(vectorizer, "MLP_Fine_Tuned/tfidf_vectorizer.pkl")
    joblib.dump(model, "MLP_Fine_Tuned/mlp_fake_news_model.pkl")
    print("Model and vectorizer saved.")

def main():
    loaded = load_and_vectorize_data("../0.2_split_data.joblib" )
    if not loaded: return
    
    (X_train_raw, y_train), (X_train_vec, X_val_vec, X_test_vec, y_validate, y_test), tfidf = loaded

    best_params = perform_grid_search(X_train_raw, y_train)

    model, history = train_manual_mlp(X_train_vec, y_train, X_test_vec, y_test, best_params)

    plot_learning_curves(history)

    evaluate_and_save(model, tfidf, X_test_vec, y_test)

if __name__ == "__main__":
    main()