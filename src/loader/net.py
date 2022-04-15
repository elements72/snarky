import tensorflow as tf
from dataclasses import dataclass, field
from loader import Loader
import argparse
from chronometer import Chronometer


@dataclass
class Snarky:
    _sequence: tf.data.Dataset
    _batch_size: int = 64
    _sequence_length: int = 25
    _params: dict = field(default_factory= lambda: {"chords": 1, "chords_play": 2,
                                                    "melody": 128, "melody_play": 2})
    _buffer_size: int = field(init=False)

    def __post_init__(self):
        self._buffer_size = self._batch_size - self._sequence_length
        self._sequence = (self._sequence.shuffle(self._buffer_size).batch(batch_size, drop_remainder=True).cache()
                          .prefetch(tf.data.experimental.AUTOTUNE))

    def net(self, lr=0.005):
        input_shape = (self._sequence_length, len(self._params))
        inputs = tf.keras.Input(input_shape)
        x = tf.keras.layers.LSTM(128)(inputs)
        outputs = {key: tf.keras.layers.Dense(self._params[key], name=key)(x) for key in self._params}

        model = tf.keras.Model(inputs, outputs)

        loss = {key: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) for key in self._params}

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(loss=loss, optimizer=optimizer)

        model.summary()
        return model

    def train(self, model, epochs=50):
        callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./training_checkpoints/ckpt_{epoch}',
                                                        save_weights_only=True),
                     tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True)]
        history = model.fit(self._sequence, epochs=epochs, callbacks=callbacks)

        return history



if __name__ == "__main__":
    with Chronometer() as t:
        parser = argparse.ArgumentParser(description='Process songs dataset.')
        parser.add_argument('path', metavar='path', type=str,
                            help='the path of the dataset')
        args = parser.parse_args()
        path = args.path
        loader = Loader(path)
        dataset = loader.load()

        batch_size = 64
        sequence_length = 25

        sequence = loader.create_sequences(dataset, sequence_length)
        params = loader.get_vocabulary()
        snarky = Snarky(_sequence=sequence, _batch_size=batch_size, _sequence_length=sequence_length, _params=params)
        net = snarky.net()
        snarky.train(net)
