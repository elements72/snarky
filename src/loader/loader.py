import os.path
from dataclasses import dataclass, field
import argparse
from chronometer import Chronometer
from Vocabulary import Vocab
import json

@dataclass
class Loader:
    path: str = "../dataset/"
    _dataset: dict = field(init=False, default_factory=lambda : {"chords": [], "chords_play": [],
                                                                 "melody": [], "melody_play": []}, repr=False)
    _chord_mapping_path: str = path + "_chord_mapping.json"

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
                print("HERE")
                mappings = json.load(fp)

        else:
            vocabulary = list(set(self._dataset["chords"]))
            for i, chord in enumerate(vocabulary):
                mappings[chord] = i
            with open(self._chord_mapping_path, "w") as fp:
                json.dump(mappings, fp, indent=4)
        return mappings


    def get_dataset(self):
        return self._dataset

if __name__ == "__main__":
    with Chronometer() as t:
        parser = argparse.ArgumentParser(description='Process songs dataset.')
        parser.add_argument('path', metavar='path', type=str,
                            help='the path of the dataset')
        parser.add_argument('-chord_mapping', type=str, required=True, help="Path of the chord mapping, if not exist is created")

        args = parser.parse_args()
        path = args.path
        chord_mapping_path = args.chord_mapping

        loader = Loader(path, chord_mapping_path)
        loader.load_dataset()
        dataset = loader.get_dataset()

        # vocab = Vocab(dataset["chords"])
        # print(list(vocab.token_to_idx.items())[:10])
        mapping = loader.load_chord_mapping()
        print(f"""Len of chords: {len(dataset['chords'])}, len of chordsPlay: {len(dataset['chords_play'])},
        len of melody: {len(dataset['melody'])}, len of melodyPlay: {len(dataset['melody_play'])}, """)
