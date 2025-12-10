from preprocessing import Preprocessing
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, MaxPooling1D, TextVectorization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix

#region Uncomment here if anything change to preprocessing
# preprocessor = Preprocessing()
# preprocessor.dump_variable(test_size=0.2)
# preprocessor.dump_variable(test_size=0.3)

X_train, X_validate, X_test, y_train, y_validate, y_test = joblib.load('0.2_split_data.joblib')

real = np.sum(y_train == 0)
fake = np.sum(y_train == 1)

labels = ['Real', 'Fake']
values = [real, fake]
plt.bar(labels, values)
for i, v in enumerate(values):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.ylabel('Count')
plt.title('Distribution of Real vs Fake')
plt.show()

plt.figure(figsize=(6, 6))
plt.pie(
    values,
    labels=labels,
    autopct='%1.1f%%',   
    startangle=90        
)

plt.show()

print(f"Training set size: {len(X_train)}"
      f"\nValidation set size: {len(X_validate)}"
      f"\nTest set size: {len(X_test)}")

print(f"Training labels distribution: {np.bincount(y_train)}"
      f"\nValidation labels distribution: {np.bincount(y_validate)}"
        f"\nTest labels distribution: {np.bincount(y_test)}")


vocab_size = 5000
max_length = 500

vectorizer = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=max_length,
)
vectorizer.adapt(X_train)

def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(1,), dtype="string"))
    model.add(vectorizer)
    
    hp_embed_dim = hp.Int('embedding_dim', min_value=32, max_value=128, step=32)
    model.add(Embedding(input_dim=vocab_size, output_dim=hp_embed_dim, mask_zero=False))
    
    # Add Dropout after embedding
    model.add(Dropout(0.3))
    
    hp_filters = hp.Int('filters', min_value=32, max_value=64, step=16)  # Reduced
    hp_kernel = hp.Choice('kernel_size', values=[3, 5])
    
    for i in range(hp.Int('num_conv_layers', 1, 3)):
        model.add(Conv1D(
            filters=hp.Int(f'filters_{i}', 32, 128, step=32),
            kernel_size=hp.Choice(f'kernel_{i}', [3, 5]),
            activation='relu',
            padding='same'
        ))
        if i < hp.Int('num_conv_layers', 1, 3) - 1:
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))
    
    model.add(GlobalMaxPooling1D())
    
    # Add Dropout before dense layer
    model.add(Dropout(0.4))
    
    hp_units = hp.Int('dense_units', min_value=32, max_value=64, step=16)  # Reduced
    model.add(Dense(units=hp_units, activation='relu'))
    
    hp_dropout = hp.Float('dropout', min_value=0.4, max_value=0.6, step=0.1)  # Higher dropout
    model.add(Dropout(rate=hp_dropout))
    
    model.add(Dense(1, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
    
    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='my_tuning_dir', # Folder where results are saved
    project_name='fake_news_cnn',

    #region FOR NEW TUNING uncomment this, (commented reason: time consuming)
    # overwrite=True 
)

# EarlyStopping to save time during the search
stop_early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

tuner.search(
    X_train, 
    y_train, 
    epochs=10, 
    callbacks=[stop_early],
    validation_data=(X_validate, y_validate),
    verbose=1
)

# --- 1. Retrain the Best Model to get the Plotting History ---
print("Retraining the best model to capture training curves...")

# Build a fresh model using the best hyperparameters found
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain it (This ensures we have the 'history' object for plotting)
# We use a slightly higher epoch count or the same as the search
history = hypermodel.fit(
    X_train, 
    y_train,
    epochs=10,
    validation_data=(X_validate, y_validate),
    callbacks=[stop_early]
)

# --- 2. Plot Training & Validation Curves ---
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_history(history)

# --- 3. Plot Confusion Matrix ---
# Get predictions (probabilities)
y_pred_probs = hypermodel.predict(X_test)

#region NOTICE HERE(OUTPUT, sigmoid activation function as last layer)
# Convert probabilities to 0 or 1 (Threshold = 0.5)
y_pred = (y_pred_probs > 0.5).astype(int)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Fake', 'Real'], 
            yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))