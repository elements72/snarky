import os.path
from dataclasses import dataclass, field
import argparse
from chronometer import Chronometer
from .Vocabulary import Vocab
import json
import tensorflow as tf
import numpy as np

@dataclass
class Loader:
    path: str = "../dataset/"
    _dataset: dict = field(init=False, default_factory=lambda: {"chords": [], "chords_play": [],
                                                                "melody": [], "melody_play": []}, repr=False)
    _vocabulary: dict = field(init=False, default_factory=lambda: {"chords": None, "chords_play": None,
                                                                   "melody": None, "melody_play": None}, repr=False)
    _chord_mapping_path: str = path + "_chord_mapping.json"
    _key_order = ["chords", "chords_play", "melody", "melody_play"]

    def add_song(self, chords: str, chords_play: str, melody: str, melody_plays: str):
        self._dataset["chords"].extend(chords.split(" "))
        self._dataset["chords_play"].extend(chords_play.split(" "))
        self._dataset["melody"].extend(melody.split(" "))
        self._dataset["melody_play"].extend(melody_plays.split(" "))

    def load_dataset(self):
        processed_song = 0
        with open(self.path, "r") as fp:
            while True:
                chords = fp.readline().strip()
                # check the end of the file
                if not chords:
                    break
                chords_play = fp.readline().strip()
                melody = fp.readline().strip()
                melody_play = fp.readline().strip()
                self.add_song(chords, chords_play, melody, melody_play)
                processed_song += 1
        print("Processed songs: ", processed_song)

    def load_chord_mapping(self):
        mappings = {}
        if os.path.isfile(self._chord_mapping_path):
            with open(self._chord_mapping_path, "r") as fp:
                mappings = json.load(fp)
        else:
            vocabulary = list(set(self._dataset["chords"]))
            for i, chord in enumerate(vocabulary):
                mappings[chord] = i
            with open(self._chord_mapping_path, "w") as fp:
                json.dump(mappings, fp, indent=4)
        return mappings

    def create_vocabulary(self) -> None:
        for key in self._dataset:
            self._vocabulary[key] = Vocab(self._dataset[key])

    def create_dataset(self) -> tf.data.Dataset:
        train_dataset = np.stack([self._vocabulary[key][self._dataset[key]] for key in self._dataset], axis=1)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        return train_dataset

    def create_sequences(self, dataset: tf.data.Dataset, seq_length: int) -> tf.data.Dataset:
        """Returns TF Dataset of sequence and label examples."""
        seq_length = seq_length + 1

        # Take 1 extra for the labels
        windows = dataset.window(seq_length, shift=1, stride=1,
                                 drop_remainder=True)

        # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
        flatten = lambda x: x.batch(seq_length, drop_remainder=True)
        sequences = windows.flat_map(flatten)

        # Split the labels
        def split_labels(sequences):
            inputs = sequences[:-1]
            labels_dense = sequences[-1]
            labels = {key: labels_dense[i] for i, key in enumerate(self._key_order)}
            return inputs, labels

        return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

    def one_hot_encoding(self, value: str):
        print("In one hot: ", value)
        #value = vocab[value]
        #encoded = tf.one_hot(value, len(vocab))
        #print(encoded)
        return value

    def load(self) -> tf.data.Dataset:
        self.load_dataset()
        self.create_vocabulary()
        return self.create_dataset()

    def get_vocabulary(self):
        return {key: len(self._vocabulary[key]) for key in self._vocabulary}

if __name__ == "__main__":
    with Chronometer() as t:
        parser = argparse.ArgumentParser(description='Process songs dataset.')
        parser.add_argument('path', metavar='path', type=str,
                            help='the path of the dataset')
        args = parser.parse_args()
        path = args.path
        loader = Loader(path)
        dataset = loader.load()
        sequence = loader.create_sequences(dataset, 25)
        for seq, target in sequence.take(1):
            print('sequence shape:', seq.shape)
            print('sequence elements (first 10):', seq[0: 10])
            print()
            print('target:', target)
        print(sequence.element_spec)

        #print(f"""Len of chords: {len(dataset['chords'])}, len of chordsPlay: {len(dataset['chords_play'])},
        #len of melody: {len(dataset['melody'])}, len of melodyPlay: {len(dataset['melody_play'])}, """)
