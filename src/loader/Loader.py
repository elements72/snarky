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
    _params: list[str] = field(default_factory=lambda: ["chords", "chords_play", "melody", "melody_play"])
    _dataset: dict = field(init=False)
    _vocabulary: dict[str: Vocab] = field(init=False)
    _mapping_path: str = "mapping"

    def __post_init__(self):
        self._dataset = {key: [] for key in self._params}
        self._vocabulary = {key: None for key in self._params}

    def add_song(self, data):
        for key in self._params:
            self._dataset[key].extend(data[key].split(" "))

    def load_dataset(self):
        processed_song = 0
        data = {}
        params_number = len(self._params)
        eof = False
        with open(self.path, "r") as fp:
            while True:
                for key in self._params:
                    data[key] = fp.readline().strip()
                    # check the end of the file
                    if not data[key]:
                        eof = True
                        break
                if eof:
                    break
                else:
                    self.add_song(data)
                processed_song += 1
        print("Processed songs: ", processed_song)

    # unused
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
            labels = {key: labels_dense[i] for i, key in enumerate(self._params)}
            return inputs, labels

        return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

    def one_hot_encoding(self, value: str):
        print("In one hot: ", value)
        #value = vocab[value]
        #encoded = tf.one_hot(value, len(vocab))
        #print(encoded)
        return value

    def print_vocabulary(self):
        for key in self._params:
            with open(f"{self._mapping_path}_{key}", "w") as fp:
                json.dump(self._vocabulary[key].token_to_idx, fp, indent=4)

    def save_song(self, song, path="./generated"):
        with open(path, "w") as fp:
            for i, param in enumerate(self._params):
                fp.write(" ".join(map(lambda x: str(self._vocabulary[param].to_tokens(x)), song[:, i])))
                fp.write("\n")

    def load(self) -> tf.data.Dataset:
        self.load_dataset()
        self.create_vocabulary()
        return self.create_dataset()

    def get_params(self) -> dict:
        """
        Return the name of the parameters parsed from the preprocessed file and the vocab size of this in a dict
        :return: dict like { param_name: vocab_size }
        :rtype: dict
        """
        return {key: len(self._vocabulary[key]) for key in self._vocabulary}

if __name__ == "__main__":
    with Chronometer() as t:
        parser = argparse.ArgumentParser(description='Process songs dataset.')
        parser.add_argument('path', metavar='path', type=str,
                            help='the path of the dataset')
        parser.add_argument('mapping_path', metavar='mapping_path', type=str,
                            help='the path of the dataset')
        args = parser.parse_args()
        path = args.path
        loader = Loader(path, _mapping_path=args.mapping_path)
        dataset = loader.load()
        sequence = loader.create_sequences(dataset, 25)
        for seq, target in sequence.take(1):
            print('sequence shape:', seq.shape)
            print('sequence elements (first 10):', seq[0: 10])
            print()
            print('target:', target)
        print(loader.print_vocabulary())

        #print(f"""Len of chords: {len(dataset['chords'])}, len of chordsPlay: {len(dataset['chords_play'])},
        #len of melody: {len(dataset['melody'])}, len of melodyPlay: {len(dataset['melody_play'])}, """)
