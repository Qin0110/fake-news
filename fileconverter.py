import keras

model = keras.saving.load_model('model/best_cnn_rnn_model.keras')
model.save('./LinuxEnv/best_cnn_rnn_model.keras')