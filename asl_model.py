import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ASLModel:
    """Model for ASL (keypoint sequence) -> text using CTC.

    Input: sequence of keypoint feature vectors shaped (T, F)
    Output: softmax over (vocab + blank) per time-step
    """
    def __init__(self, input_features=225, rnn_units=256, max_text_length=200):
        self.input_features = input_features
        self.rnn_units = rnn_units
        self.max_text_length = max_text_length
        self.model = None
        self.char_to_num = None
        self.num_to_char = None

    def build_model(self, vocab_size):
        inp = layers.Input(shape=(None, self.input_features), name="input")

        # Optional small dense frontend to mix channels
        x = layers.TimeDistributed(layers.Dense(256, activation="relu"))(inp)
        x = layers.Dropout(0.2)(x)

        x = layers.Bidirectional(layers.LSTM(self.rnn_units, return_sequences=True, dropout=0.2))(x)
        x = layers.Bidirectional(layers.LSTM(self.rnn_units, return_sequences=True, dropout=0.2))(x)

        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        # vocab_size + 1 for CTC blank
        out = layers.Dense(vocab_size + 1, activation="softmax", name="softmax_output")(x)

        self.model = keras.Model(inputs=inp, outputs=out, name="asl_to_text")
        return self.model

    def create_vocabulary(self, texts):
        chars = sorted(set("".join(texts)))
        self.char_to_num = {c: i for i, c in enumerate(chars)}
        self.num_to_char = {i: c for i, c in enumerate(chars)}
        return len(chars)

    def encode_text(self, text):
        return [self.char_to_num[c] for c in text if c in self.char_to_num]

    def decode_prediction(self, prediction):
        # prediction: (batch, T, vocab+1)
        batch = prediction.shape[0]
        input_len = np.ones(batch) * prediction.shape[1]
        decoded = keras.backend.ctc_decode(prediction, input_length=input_len, greedy=True)[0][0]
        texts = []
        for seq in decoded:
            chars = [self.num_to_char[int(i)] for i in seq.numpy() if int(i) != -1]
            texts.append("".join(chars))
        return texts

    @staticmethod
    def ctc_loss(y_true, y_pred):
        # y_true: dense padded labels (batch, max_label_len)
        # y_pred: (batch, T, vocab+1)
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int32")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int32")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int32")

        input_length = input_length * tf.ones(shape=(batch_len,), dtype="int32")
        label_length = label_length * tf.ones(shape=(batch_len,), dtype="int32")

        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss

    def compile_model(self, lr=1e-4):
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=self.ctc_loss)

    def preprocess_sequence(self, seq_or_path):
        """Load .npy sequence or accept an array. Returns (T, F) float32."""
        if isinstance(seq_or_path, str):
            arr = np.load(seq_or_path)
        else:
            arr = np.asarray(seq_or_path)
        return arr.astype(np.float32)

    def predict(self, seq_or_path):
        feats = self.preprocess_sequence(seq_or_path)
        feats = np.expand_dims(feats, axis=0)
        pred = self.model.predict(feats)
        return self.decode_prediction(pred)[0]

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = keras.models.load_model(path, custom_objects={'ctc_loss': self.ctc_loss})


if __name__ == '__main__':
    m = ASLModel()
    sample_texts = ["hello", "yes", "no"]
    vocab = m.create_vocabulary(sample_texts)
    m.build_model(vocab)
    m.compile_model()
    m.model.summary()
