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
    path: str = "../dataset/dataMD"
    _params: list = field(default_factory=lambda: ["chords", "chords_play", "melody", "melody_play"])
    _dataset: dict = field(init=False)
    _vocabulary: dict = field(init=False)
    _mapping_path: str = "mapping"

    def __post_init__(self):
        self._dataset = {key: [] for key in self._params}
        self._vocabulary = {key: None for key in self._params}

    def add_song(self, data):
        for key in self._params:
            self._dataset[key].extend(data[key].split(" "))

    def load_dataset(self):
        print(self.path)
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

    def create_vocabulary(self) -> None:
        for key in self._dataset:
            self._vocabulary[key] = Vocab(self._dataset[key])

    def encode_song(self, song: dict) -> dict:
        """
        Encode a single song
        :param song: Song to be encoded in a dict format
        :type song: dict[str:]
        :return: encoded song array
        """
        song = {key: self._vocabulary[key][song[key]] for key in song}
        for key in song:
            song[key] = tf.keras.utils.to_categorical(song[key], num_classes=len(self._vocabulary[key]))

        return song

    def create_datasets(self, categorical=True) -> list:
        train_datasets = [self._vocabulary[key][self._dataset[key]] for key in self._dataset]
        if categorical:
            for i, param in enumerate(self._params):
                train_datasets[i] = tf.keras.utils.to_categorical(train_datasets[i], num_classes=len(self._vocabulary[param]))
        train_datasets = [tf.data.Dataset.from_tensor_slices(dataset) for dataset in train_datasets]
        return train_datasets

    def create_dataset(self) -> tf.data.Dataset:
        train_dataset = np.stack([self._vocabulary[key][self._dataset[key]] for key in self._dataset], axis=1)
        print(train_dataset.shape)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        return train_dataset

    def create_sequences(self, datasets: list, seq_length: int, latent=False) -> tf.data.Dataset:
        """Returns TF Dataset of sequence and label examples."""
        seq_length = seq_length + 1

        # Take 1 extra for the labels
        windows = [dataset.window(seq_length, shift=1, stride=1,
                                 drop_remainder=True) for dataset in datasets]

        # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
        flatten = lambda x: x.batch(seq_length, drop_remainder=True)
        sequences = [window.flat_map(flatten) for window in windows]


        # Split the labels
        def split_label(sequences):
            inputs = sequences[:-1]
            label = sequences[-1]
            # labels = {key: labels_dense[i] for i, key in enumerate(self._params)}
            return inputs, label

        def mapping(*seqs):
            seqs = {key: split_label(sq) for key, sq in zip(self._params, seqs)}
            #chords, chords_label = split_label(seq1)
            #chords_play, chords_play_label = split_label(seq2)
            #melody, melody_label = split_label(seq3)
            #melody_play, melody_play_label = split_label(seq4)

            #inputs = (chords, chords_play, melody, melody_play)
            #labels = (chords_label, chords_play_label, melody_label, melody_play_label)
            inputs = tuple(seqs[key][0] for key in seqs)
            labels = tuple(seqs[key][1] for key in seqs)
            labels = {key: labels[i] for i, key in enumerate(self._params)}
            if latent:
                inputs = (inputs, [1 for _ in range(128)])
            return inputs, labels

        dataset = tf.data.Dataset.zip((*sequences, ))
        return dataset.map(mapping)

    def print_vocabulary(self):
        for key in self._params:
            with open(f"{self._mapping_path}_{key}", "w") as fp:
                fp.write(f"\t{key}\tindex\tfreq\t%\n")
                for idx, token in enumerate(self._vocabulary[key].token_to_idx):
                    freq = self._vocabulary[key].token_freqs[idx][1]
                    fp.write(f"\t'{token}':\t{idx}\t{freq}"
                             f"\t{(freq/self._vocabulary[key].total)*100:.2f}\n")

    def save_song(self, song, path="./generated"):
        with open(path, "w") as fp:
            for i, param in enumerate(self._params):
                fp.write(" ".join(map(lambda x: str(self._vocabulary[param].to_tokens(x)), song[:, i])))
                fp.write("\n")

    def set_vocabulary(self, vocab):
        self._vocabulary = vocab

    def load(self, vocab=None, categorical=True) -> list:
        self.load_dataset()
        if vocab is None:
            self.create_vocabulary()
        else:
            self._vocabulary = vocab
        return self.create_datasets(categorical)

    def get_vocabulary(self):
        return self._vocabulary

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
        params = ["chords", "chords_play", "melody", "melody_play"]
        loader = Loader(path, _mapping_path=args.mapping_path, _params=params)
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
