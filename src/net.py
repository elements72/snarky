import tensorflow as tf
from dataclasses import dataclass, field
from loader import Loader
import argparse
from chronometer import Chronometer
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
        self._sequence = (self._sequence.shuffle(self._buffer_size).batch(self._batch_size, drop_remainder=True).cache()
                          .prefetch(tf.data.experimental.AUTOTUNE))

    def create_model(self, lr=0.005):
        input_shape = (self._sequence_length, len(self._params))
        inputs = tf.keras.Input(input_shape)
        x = tf.keras.layers.LSTM(128)(inputs)
        outputs = {key: tf.keras.layers.Dense(self._params[key], name=key)(x) for key in self._params}

        model = tf.keras.Model(inputs, outputs)

        loss = {key: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) for key in self._params}

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(loss=loss, optimizer=optimizer)

        # model.summary()

        self.model = model

        return self.model

    def train(self, epochs=50):
        callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./training_checkpoints/ckpt_{epoch}',
                                                        save_weights_only=True),
                     tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True)]
        history = self.model.fit(self._sequence, epochs=epochs, callbacks=callbacks)

        return history

    def evaluate(self):
        return self.model.evaluate(self._sequence, return_dict=True)

    def predict_next_note(self, notes, temperature: float = 1.0):
        assert temperature > 0

        inputs = tf.expand_dims(notes, 0)
        predictions = self.model.predict(inputs)
        melody_logits = predictions['melody']
        chords_logits = predictions['chords']
        chord_play_logits = predictions['chords_play']
        melody_play_logits = predictions['melody_play']

        melody_logits /= temperature
        melody = tf.random.categorical(melody_logits, num_samples=1)
        melody = tf.squeeze(melody, axis=-1)

        chords_logits /= temperature
        chord = tf.random.categorical(chords_logits, num_samples=1)
        chord = tf.squeeze(chord, axis=-1)

        melody_play = tf.random.categorical(melody_play_logits, num_samples=1)
        melody_play = tf.squeeze(melody_play, axis=-1)
        chord_play = tf.random.categorical(chord_play_logits, num_samples=1)
        chord_play = tf.squeeze(chord_play, axis=-1)

        return int(chord), int(chord_play), int(melody), int(melody_play)

    def generate(self, inputs, temperature: float = 0.1, num_predictions: int = 125):
        generated_notes = []
        for _ in range(num_predictions):
            chord, chord_play, note, note_play = self.predict_next_note(inputs, temperature)
            generated = (chord, chord_play, note, note_play)
            generated_notes.append(generated)
            inputs = np.delete(inputs, 0, axis=0)
            inputs = np.append(inputs, np.expand_dims(generated, 0), axis=0)
        return np.array(generated_notes)




if __name__ == "__main__":
    with Chronometer() as t:
        parser = argparse.ArgumentParser(description='Process songs dataset.')
        parser.add_argument('path', metavar='path', type=str,
                            help='the path of the dataset')
        parser.add_argument('-t', metavar='temperature', type=float, help='temperature value', required=False)
        parser.add_argument('-predictions', metavar='temperature', type=int, help='temperature value', required=False)
        parser.add_argument('-source', metavar='source', type=str, help='source melody', required=False)
        args = parser.parse_args()
        path = args.path
        source_melody = args.source
        print(source_melody)

        loader = Loader(path)
        dataset = loader.load()

        batch_size = 64
        sequence_length = 25
        num_predictions = args.predictions if args.predictions else 125

        sequence = loader.create_sequences(dataset, sequence_length)
        params = loader.get_params()
        snarky = Snarky(_sequence=sequence, _batch_size=batch_size, _sequence_length=sequence_length, _params=params)
        snarky.create_model()
        for seq, _ in sequence.take(1):
            generated = snarky.generate(seq)
            loader.save_song(generated)


        # snarky.train(net)
