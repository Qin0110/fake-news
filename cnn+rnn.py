# Architecture reference:
# https://www.sciencedirect.com/science/article/pii/S2667096820300070?via%3Dihub

import os

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

import keras
import keras_tuner as kt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (Conv1D, Dense, Dropout, Embedding, Input, LSTM,
                          MaxPooling1D, TextVectorization)
from keras.models import Sequential
from keras.optimizers import Adam

# Where every artefact of this run is written.
MODEL_DIR = "model"
SPLIT = 0.2                       # matches '<SPLIT>_split_data.joblib'
DATA_FILE = f"{SPLIT}_split_data.joblib"


class Model_Building:
    def __init__(self, vocab_size=5000, max_length=500, data_file=DATA_FILE,
                 model_dir=MODEL_DIR):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.data_file = data_file
        self.model_dir = model_dir
        self.tag = f"cnn_rnn_{SPLIT}_split"

        os.makedirs(self.model_dir, exist_ok=True)

        (self.X_train, self.X_validate, self.X_test,
         self.y_train, self.y_validate, self.y_test) = self.dataset_preparation()

        self.vectorizer = TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.max_length,
        )
        self.vectorizer.adapt(self.X_train)

    def dataset_preparation(self):
        # Uncomment if preprocessing changes:
        # from preprocessing import Preprocessing
        # Preprocessing().dump_variable(test_size=0.2)

        if not os.path.exists(self.data_file):
            raise FileNotFoundError(
                f"'{self.data_file}' not found. Rebuild the splits with "
                "`python preprocessing.py` (requires fake_news.csv)."
            )

        X_train, X_validate, X_test, y_train, y_validate, y_test = joblib.load(
            self.data_file
        )
        print(f"Loaded {self.data_file}: train={len(X_train)}, "
              f"validate={len(X_validate)}, test={len(X_test)}")
        return X_train, X_validate, X_test, y_train, y_validate, y_test

    def build_model(self, hp):
        model = Sequential()
        model.add(Input(shape=(1,), dtype="string"))
        model.add(self.vectorizer)

        hp_embedding_dim = hp.Choice(
            'embedding_dim', values=[50, 64, 100, 128, 200, 256, 300], default=100
        )
        model.add(Embedding(input_dim=self.vocab_size, output_dim=hp_embedding_dim))
        model.add(Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1)))

        # Resolve depth once. Re-reading hp.Int inside the loop is what left the
        # pooling layer outside the loop body, so only one was ever added.
        num_conv_layers = hp.Int('num_conv_layers', 1, 3)
        for i in range(num_conv_layers):
            model.add(Conv1D(
                filters=hp.Int(f'filters_{i}', 32, 128, step=32),
                kernel_size=hp.Choice(f'kernel_{i}', [3, 5]),
                activation='relu',
                padding='same'
            ))
            # Never pool after the last block: the LSTM needs sequence length.
            if i < num_conv_layers - 1:
                model.add(MaxPooling1D(pool_size=2))

        model.add(LSTM(
            units=hp.Int('lstm_units', 32, 128, step=32),
            return_sequences=False,
            activation=hp.Choice('lstm_activation', values=['tanh', 'relu'],
                                 default='tanh')
        ))
        model.add(Dense(
            units=hp.Int('dense_units', 32, 128, step=32),
            activation=hp.Choice('dense_activation', values=['relu', 'tanh'],
                                 default='relu')
        ))
        model.add(Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1)))
        model.add(Dense(1, activation='sigmoid'))

        hp_learning_rate = hp.Choice(
            'learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5], default=1e-3
        )
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

        stop_early = EarlyStopping(monitor='val_loss', patience=3,
                                   restore_best_weights=True, mode='min', verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3,
                                      min_lr=1e-6, verbose=1)

        # No `epochs=` here: it would override Hyperband's bracket schedule.
        tuner.search(
            self.X_train,
            self.y_train,
            validation_data=(self.X_validate, self.y_validate),
            callbacks=[stop_early, reduce_lr]
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("\nBest hyperparameters:")
        for name, value in best_hps.values.items():
            print(f"  {name}: {value}")

        return tuner.hypermodel.build(best_hps)

    def model_training(self, model):
        checkpoint_path = os.path.join(self.model_dir, 'best_cnn_rnn_model.keras')

        stop_early = EarlyStopping(monitor='val_loss', patience=10,
                                   restore_best_weights=True, mode='min', verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3,
                                      min_lr=1e-6, verbose=1)
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                     save_best_only=True, mode='min', verbose=1)

        history = model.fit(
            self.X_train,
            self.y_train,
            epochs=100,
            validation_data=(self.X_validate, self.y_validate),
            callbacks=[stop_early, reduce_lr, checkpoint]
        )

        # EarlyStopping restored best weights, so this agrees with the checkpoint.
        keras.saving.save_model(model, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        return history

    def evaluate_model(self, model):
        y_pred_prob = model.predict(self.X_test).reshape(-1)
        y_pred = (y_pred_prob >= 0.5).astype(int)

        report = classification_report(self.y_test, y_pred,
                                       target_names=['Real', 'Fake'])
        summary = (
            f"Accuracy : {accuracy_score(self.y_test, y_pred):.4f}\n"
            f"Precision: {precision_score(self.y_test, y_pred):.4f}\n"
            f"Recall   : {recall_score(self.y_test, y_pred):.4f}\n"
            f"F1 score : {f1_score(self.y_test, y_pred):.4f}\n"
            # Probabilities, not 0/1 labels: those collapse the curve to 3 points.
            f"ROC AUC  : {roc_auc_score(self.y_test, y_pred_prob):.4f}\n"
        )
        print("\n" + summary)
        print("Classification Report:\n", report)

        report_path = os.path.join(
            self.model_dir, f'{self.tag}_classification_report.txt'
        )
        with open(report_path, 'w', encoding='utf-8') as handle:
            handle.write(summary + "\n" + report)
        print(f"Report saved to {report_path}")

        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        cm_path = os.path.join(self.model_dir, f'{self.tag}_confusion_matrix.png')
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        plt.show()

    def plot_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, 'bo-', label='Training Accuracy')
        plt.plot(epochs_range, val_acc, 'ro-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, 'bo-', label='Training Loss')
        plt.plot(epochs_range, val_loss, 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        curve_path = os.path.join(
            self.model_dir, f'{self.tag}_training_validation_curves.png'
        )
        plt.savefig(curve_path)
        print(f"Curves saved to {curve_path}")
        plt.show()


if __name__ == "__main__":
    builder = Model_Building()
    best_model = builder.model_tuning()
    training_history = builder.model_training(best_model)
    builder.evaluate_model(best_model)
    builder.plot_history(training_history)
