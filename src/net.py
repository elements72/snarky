import tensorflow as tf
from tensorflow import keras
from dataclasses import dataclass, field
import numpy as np
from keras import backend as K


@dataclass
class Snarky:
    _sequence: tf.data.Dataset
    _batch_size: int = 64
    _sequence_length: int = 25
    _params: dict = field(default_factory= lambda: {"chords": 1, "chords_play": 2,
                                                    "melody": 128, "melody_play": 2})
    _buffer_size: int = field(init=False)
    model: tf.keras.Model = field(init=False, repr=False)

    def __post_init__(self):
        self._buffer_size = self._batch_size - self._sequence_length
        self._sequence = (self._sequence.shuffle(self._buffer_size).batch(self._batch_size, drop_remainder=True)
                          .prefetch(tf.data.experimental.AUTOTUNE))
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


    def plot_model(self, to="model.png"):
        tf.keras.utils.plot_model(
            self.model,
            to_file=to,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True
        )

    def summary(self):
        self.model.summary()

    def create_autoencoder(self, line, num_feature: int, num_units=128):
        encoder = tf.keras.layers.LSTM(num_units, activation='relu', return_sequences=True)(line)
        encoder = tf.keras.layers.LSTM(int(num_units/2), activation='relu', return_sequences=False)(encoder)

        repeat_vector = tf.keras.layers.RepeatVector(self._sequence_length)(encoder)

        decoder = tf.keras.layers.LSTM(int(num_units/2), activation='relu', return_sequences=True)(repeat_vector)
        decoder = tf.keras.layers.LSTM(num_units, activation='relu', return_sequences=True)(decoder)

        return tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_feature))(decoder)

    def autoencoder(self, lr=0.001, num_units=128):
        inputs = []
        autoencoder_out = []
        for i, param in enumerate(self._params):
            num_feature = self._params[param]
            line = tf.keras.Input((self._sequence_length, self._params[param]), name=param)
            inputs.append(line)
            encoder = tf.keras.layers.LSTM(num_units, activation='relu', return_sequences=True)(line)
            encoder = tf.keras.layers.LSTM(int(num_units / 2), activation='relu', return_sequences=False)(encoder)

            repeat_vector = tf.keras.layers.RepeatVector(self._sequence_length)(encoder)

            decoder = tf.keras.layers.LSTM(int(num_units / 2), activation='relu', return_sequences=True)(repeat_vector)
            decoder = tf.keras.layers.LSTM(num_units, activation='relu', return_sequences=True)(decoder)
            time_dist = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_feature))(decoder)
            autoencoder_out.append(time_dist)
        concat = tf.keras.layers.Concatenate()(autoencoder_out)
        x = tf.keras.layers.LSTM(num_units)(concat)

        outputs = {key: tf.keras.layers.Dense(self._params[key], name=f"{key}_output", activation="softmax")(x)
                   for key in self._params}

        model = tf.keras.Model(inputs, outputs)

        loss = {key: tf.keras.losses.CategoricalCrossentropy() for key in self._params}
        metrics = {key: tf.keras.metrics.CategoricalAccuracy() for key in self._params}

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        model.summary()
        self.model = model

        return self.model


    def vae(self, num_units=128, time_step=8, lr=0.001):
        latent_d = 128
        conductor_d = 128
        inputs = [tf.keras.Input((self._sequence_length, self._params[param]), name=param) for param in self._params]
        #z_constant = tf.zeros((self._ba,latent_d))
        #z_constant = K.variable(z_constant)
        #z = tf.keras.Input(latent_d, tensor=z_constant)
        z = tf.keras.Input(latent_d)
        concat = tf.keras.layers.Concatenate(axis=-1)(inputs)
        z0 = z[0]

        # conductor
        conductor_seq_len = self._sequence_length//time_step
        print(self._batch_size, conductor_seq_len, z[0])
        conductor = tf.keras.layers.LSTM(latent_d, stateful=False, return_sequences=True) \
            (tf.zeros((self._batch_size, conductor_seq_len, 1)), initial_state=[z, z])

        expanded_conductor = []
        print(conductor)
        # bar generator
        for i in range(0, conductor_seq_len):
            tmp = tf.keras.layers.RepeatVector(time_step)(conductor[:, i])
            expanded_conductor.append(tmp)
        concat_conductor = tf.keras.layers.Concatenate(axis=1)(expanded_conductor)
        print("Concat conductor: ", concat_conductor)
        lstm_input = tf.keras.layers.Concatenate(axis=-1)([concat_conductor, concat])

        x = tf.keras.layers.LSTM(num_units)(lstm_input)
        outputs = {key: tf.keras.layers.Dense(self._params[key], name=f"{key}_output", activation="softmax")(x) for key
                   in self._params}

        model = tf.keras.Model([inputs, z], outputs)
        loss = {key: tf.keras.losses.CategoricalCrossentropy() for key in self._params}
        metrics = {key: tf.keras.metrics.CategoricalAccuracy() for key in self._params}

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # model.summary()
        self.model = model

        return self.model


    def create_model(self, lr=0.001, num_units=128):
        inputs = [tf.keras.Input((self._sequence_length, self._params[param]), name=param) for param in self._params]
        concat = tf.keras.layers.Concatenate(axis=-1)(inputs)

        x = tf.keras.layers.LSTM(num_units)(concat)

        outputs = {key: tf.keras.layers.Dense(self._params[key], name=f"{key}_output", activation="softmax")(x) for key in self._params}

        model = tf.keras.Model(inputs, outputs)

        loss = {key: tf.keras.losses.CategoricalCrossentropy() for key in self._params}
        metrics = {key: tf.keras.metrics.CategoricalAccuracy() for key in self._params}

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # model.summary()
        self.model = model

        return self.model

    def train(self, epochs=50):
        callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./training_checkpoints/ckpt_{epoch}',
                                                        save_weights_only=True),
                     tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True)]
        history = self.model.fit(self._sequence, epochs=epochs, callbacks=callbacks)

        return history

    def evaluate(self, sequence: tf.data.Dataset):
        return self.model.evaluate(x=sequence, return_dict=True)

    def predict_next_note(self, inputs: list, temperature: float = 1.0):
        assert temperature > 0
        inputs = [tf.expand_dims(line, 0) for line in inputs]
        predictions = self.model.predict(inputs)
        # predicted = [int(tf.squeeze(tf.argmax(predictions[param], axis=-1), axis=-1)) for param in self._params]
        print("Before")
        predicted = [int(tf.squeeze(tf.random.categorical(predictions[param], num_samples=1), axis=-1)) for param in self._params]
        print("After: ", predicted)
        return tuple(predicted)

    def save(self, path="model.params") -> None:
        """
        Save the net parameters
        """

        self.model.save_weights(path)

    def load(self, path="model.params"):
        """
        Load the net parameters
        :return:
        :rtype:
        """
        self.model.load_weights(path)

    def generate(self, inputs, temperature: float = 1, num_predictions: int = 125):
        generated_notes = []
        inputs = [inputs[key][:self._sequence_length] for key in inputs]
        for i in range(num_predictions):
            generated = self.predict_next_note(inputs, temperature)
            generated_notes.append(generated)
            # For each line delete and append the new the prediction
            inputs = [np.delete(line, 0, axis=0) for line in inputs]
            inputs = [np.append(line, np.expand_dims(tf.keras.utils.to_categorical(predicted, num_classes=self._params[label]), 0), axis=0)
                      for line, predicted, label in zip(inputs, generated, self._params)]
        return np.array(generated_notes)
