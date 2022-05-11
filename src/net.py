import tensorflow as tf
from dataclasses import dataclass, field
import numpy as np


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

    def create_model2(self, lr=0.001):
        input_shape = (self._sequence_length, len(self._params))

        #input_chord = tf.keras.Input((self._sequence_length, self._params["chords"]))
        #input_chord_play = tf.keras.Input((self._sequence_length, self._params["chords_play"]))
        #input_melody = tf.keras.Input((self._sequence_length, self._params["melody"]))
        #input_melody_play = tf.keras.Input((self._sequence_length, self._params["melody_play"]))


        inputs = [tf.keras.Input((self._sequence_length, self._params[param])) for param in self._params]
        #inputs = tf.keras.layers.Concatenate(axis=-1)([input_chord, input_chord_play, input_melody, input_melody_play])
        concat = tf.keras.layers.Concatenate(axis=-1)(inputs)

        x = tf.keras.layers.LSTM(128)(concat)

        outputs = {key: tf.keras.layers.Dense(self._params[key], name=key)(x) for key in self._params}

        model = tf.keras.Model(inputs, outputs)

        loss = {key: tf.keras.losses.CategoricalCrossentropy(from_logits=True) for key in self._params}
        metrics = {key: tf.keras.metrics.CategoricalAccuracy() for key in self._params}

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        model.summary()
        self.model = model

        return self.model
    def create_model(self, lr=0.001):
        input_shape = (self._sequence_length, len(self._params))
        inputs = tf.keras.Input(input_shape)
        x = tf.keras.layers.LSTM(128)(inputs)
        outputs = {key: tf.keras.layers.Dense(self._params[key], name=key)(x) for key in self._params}

        model = tf.keras.Model(inputs, outputs)

        loss = {key: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) for key in self._params}
        metrics = {key: tf.keras.metrics.SparseCategoricalAccuracy() for key in self._params}

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        model.summary()
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
        predicted = [int(tf.squeeze(tf.argmax(predictions[param], axis=-1), axis=-1)) for param in self._params]
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
            print("Generated: ", generated)
            generated_notes.append(generated)
            # For each line delete and append the new the prediction
            inputs = [np.delete(line, 0, axis=0) for line in inputs]
            inputs = [np.append(line, np.expand_dims(tf.keras.utils.to_categorical(predicted, num_classes=self._params[label]), 0), axis=0)
                      for line, predicted, label in zip(inputs, generated, self._params)]
        return np.array(generated_notes)
