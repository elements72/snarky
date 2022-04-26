import music21 as m21
from dataclasses import dataclass, field
import argparse


@dataclass
class Decoder:
    stream: m21.stream = field(init=False, default_factory=lambda: m21.stream.Stream())
    time_step = 0.125
    offset: float = 0

    def encode_chord(self, chord, duration):
        print(chord)
        if chord == "NC":
            chord = m21.harmony.NoChord()
        else:
            chord = chord.split(",")[0]
            chord = m21.harmony.ChordSymbol(chord)
        chord.duration.quarterLength = duration * self.time_step
        print(chord)
        self.stream.append(chord)

    def encode_note(self, note, duration):
        print(f"Adding note: {note} with duration {duration}")
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
        for symbol, rep in zip(symbols, symbols_play):
            # print(chord)
            # if repetition symbol is one we got a new chord
            if rep == "1":
                encoding_function(last_symbol, duration)
                duration = 1
            else:
                duration += 1
            last_symbol = symbol

    def create_midi(self, path="generated"):
        with open(path, "r") as fp:
            self.encode(fp, self.encode_chord)
            self.encode(fp, self.encode_note)
        self.stream.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process songs dataset.')
    parser.add_argument('path', metavar='path', type=str,
                        help='the path of the song')
    args = parser.parse_args()
    path = args.path
    print(path)
    dec = Decoder()

    dec.create_midi(path=path)
