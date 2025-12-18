#region REFERECES
# References of architecture building: https://www.sciencedirect.com/science/article/pii/S2667096820300070?via%3Dihub

from preprocessing import Preprocessing

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import keras
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, TextVectorization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

class Model_Building:
    def __init__(self, vocab_size = 5000, max_length = 500):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.X_train, self.X_validate, self.X_test, self.y_train, self.y_validate, self.y_test = self.dataset_preparation()
        self.vectorizer = TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.max_length,
        )
        self.vectorizer.adapt(self.X_train)

    def dataset_preparation(self):
        #region Uncomment here if anything change to preprocessing
        # preprocessor = Preprocessing()
        # preprocessor.dump_variable(test_size=0.2)
        # preprocessor.dump_variable(test_size=0.3)

        X_train, X_validate, X_test, y_train, y_validate, y_test = joblib.load('0.2_split_data.joblib')

        return X_train, X_validate, X_test, y_train, y_validate, y_test
    
    def build_model(self, hp):
        model = Sequential()
        model.add(Input(shape=(1,), dtype="string"))
        model.add(self.vectorizer)

        hp_embedding_dim = hp.Choice('embedding_dim', values=[50, 64, 100, 128, 200, 256, 300], default=100)
        model.add(Embedding(input_dim=self.vocab_size, output_dim=hp_embedding_dim))
        model.add(Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1)))
        for i in range(hp.Int('num_conv_layers', 1, 3)):
            model.add(Conv1D(
            filters=hp.Int(f'filters_{i}', 32, 128, step=32),
            kernel_size=hp.Choice(f'kernel_{i}', [3, 5]),
            activation='relu',
            padding='same'
        ))
        if i < hp.Int('num_conv_layers', 1, 3) - 1:
            model.add(MaxPooling1D(pool_size=2))
            # model.add(Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1)))
        model.add(LSTM(
            units=hp.Int('lstm_units', 32, 128, step=32), return_sequences=False, 
            activation=hp.Choice('lstm_activation', values=['tanh', 'relu'], default='tanh')
        ))
        model.add(Dense(
            units=hp.Int('dense_units', 32, 128, step=32),
            activation=hp.Choice('dense_activation', values=['relu', 'tanh'], default='relu')
        ))
        model.add(Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1)))
        model.add(Dense(1, activation='sigmoid'))
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5], default=1e-3)
        model.compile(
            optimizer=Adam(learning_rate=hp_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def model_tuning(self):
        tuner = kt.Hyperband(
            self.build_model,
            objective='val_accuracy',
            max_epochs=50,
            factor=3,
            directory='model_tuning',
            project_name='fake_news_cnn',
            # overwrite=True 
        )

        stop_early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, mode='min', verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
        

        tuner.search(
            self.X_train, 
            self.y_train, 
            epochs=50, 
            validation_data=(self.X_validate, self.y_validate), 
            callbacks=[stop_early, reduce_lr]
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)

        return best_model
    
    def model_training(self, model):
        stop_early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min', verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
        checkpoint = ModelCheckpoint('model/best_cnn_rnn_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

        history = model.fit(
            self.X_train, 
            self.y_train, 
            epochs=100, 
            validation_data=(self.X_validate, self.y_validate), 
            callbacks=[stop_early, reduce_lr, checkpoint]
        )

        keras.saving.save_model(model, './model/best_cnn_rnn_model.keras')
        return history
    
    def evaluate_model(self, model):
        y_pred_prob = model.predict(self.X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("\nClassification Report:\n", classification_report(self.y_test, y_pred))

        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    
    def plot_history(self, history):
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
        plt.savefig('./model/training_validation_curves.png')
        plt.show()


if __name__ == "__main__":
    model = Model_Building()
    best_model = model.model_tuning()
    history = model.model_training(best_model)
    model.evaluate_model(best_model)
    model.plot_history(history)