import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import keras
from keras import layers
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import joblib

X_train, X_validate, X_test, y_train, y_validate, y_test = joblib.load('0.3_split_data.joblib')


tfidf = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf.fit_transform(X_train).toarray() 
X_val_vec = tfidf.transform(X_validate).toarray()
X_test_vec = tfidf.transform(X_test).toarray()

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train_vec.shape[1],))) 
    
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=128, step=32),
            activation=hp.Choice('activation', ['relu', 'tanh'])
        ))
        model.add(layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    learning_rate = hp.Choice('learning_rate', [0.01, 0.001, 0.0001])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

print("Starting Keras Tuner...")
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    project_name='fake_news_mlp_fixed'
)

stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

tuner.search(X_train_vec, y_train, epochs=10, validation_data=(X_val_vec, y_validate), callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. 
Optimal Learning Rate: {best_hps.get('learning_rate')}
Optimal Layers: {best_hps.get('num_layers')}
""")

print("Retraining best model to capture full history...")
model = tuner.hypermodel.build(best_hps)


history = model.fit(
    X_train_vec, y_train,         
    epochs=20,
    validation_data=(X_test_vec, y_test), 
    verbose=1
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', linestyle='--')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss', linestyle='--')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (Binary Crossentropy)')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

model.save("fake_news_model.h5")