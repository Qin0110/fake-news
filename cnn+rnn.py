from preprocessing import Preprocessing

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

        X_train, X_validate, X_test, y_train, y_validate, y_test = joblib.load('0.3_split_data.joblib')

        return X_train, X_validate, X_test, y_train, y_validate, y_test
    
    def build_model(self, hp):
        model = Sequential()
        model.add(Input(shape=(1,), dtype="string"))
        model.add(self.vectorizer)
        
        model.add(Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1)))

        hp_embedding_dim = hp.Choice('embedding_dim', values=[50, 64, 100, 128, 200, 256, 300], default=100)
        model.add(Embedding(input_dim=self.vocab_size, output_dim=hp_embedding_dim))
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
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4], default=1e-3)
        model.compile(
            optimizer=Adam(learning_rate=hp_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model