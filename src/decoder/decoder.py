import os.path
import pathlib

import music21 as m21
from dataclasses import dataclass, field
import argparse


@dataclass
class Decoder:
    stream: m21.stream = field(init=False, default_factory=lambda: m21.stream.Stream())
    time_step: float = 0.125
    offset: float = 0

    def encode_chord(self, chord, duration):
        if chord == "NC":
            chord = m21.harmony.NoChord()
        else:
            chord = chord.split(",")[0]
            chord = m21.harmony.ChordSymbol(chord)
        chord.duration.quarterLength = duration * self.time_step
        self.stream.append(chord)

    def encode_note(self, note, duration):
        # print(f"Adding note: {note} with duration {duration}")
        if note == "r":
            note = m21.note.Rest()
        else:
            note = m21.note.Note(int(note))
        note.duration.quarterLength = duration * self.time_step
        self.stream.insert(self.offset, note)
        self.offset = self.offset + note.duration.quarterLength

    def encode(self, fp, encoding_function):
        symbols = fp.readline()
        symbols_play = fp.readline()
        symbols = symbols.split(" ")
        symbols_play = symbols_play.split(" ")

        last_symbol = symbols[0]
        duration = 1
        for symbol, rep in zip(symbols[1:], symbols_play[1:]):
            # print(chord)
            # if repetition symbol is one we got a new chord
            if rep == "1" or last_symbol != symbol:
                encoding_function(last_symbol, duration)
                duration = 1
            else:
                duration += 1
            last_symbol = symbol

    def show_midi(self, path="generated", name="generated"):
        self.stream.insert(0, m21.metadata.Metadata(title=name))
        with open(path, "r") as fp:
            self.encode(fp, self.encode_chord)
            self.encode(fp, self.encode_note)
        self.stream.show()

    def save_midi(self, path="generated", name="generated", source_melody="generated"):
        dir = os.path.join(path, "midi")
        self.stream.insert(0, m21.metadata.Metadata(title=name))
        with open(os.path.join(path, name), "r") as fp:
            self.encode(fp, self.encode_chord)
            self.encode(fp, self.encode_note)
        if not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.exists(os.path.join(dir, source_melody)):
            os.makedirs(os.path.join(dir, source_melody))
        self.stream.write('mxl', fp=os.path.join(dir, source_melody, name + ".mxl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process songs dataset.')
    parser.add_argument('path', metavar='path', type=str,
                        help='the path of the song')
    parser.add_argument('-time_step', metavar='path', type=float,
                        help='Time step')
    args = parser.parse_args()
    path = args.path
    dec = Decoder(time_step=args.time_step)
    name = pathlib.PurePath(path).name
    dec.show_midi(path=path, name=name)
