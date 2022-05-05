import fractions
import os
import music21 as m21
from .transposer import Transposer
from .song import Song
import argparse
from chronometer import Chronometer
from .scanner import Scanner

class Preprocessor:
    """_summary_
    Preprocess data 
    """
    def __init__(self, dataset_path, save_path="./dataset", time_step=0.125) -> None:
        self.dataset_path = dataset_path
        self.supportedFormats = [".mid", ".krn", ".mxl"]
        self.tr = Transposer()
        self.savePath = save_path
        self._noChordSymbol = "NC"
        self._time_step = time_step

    def load_songs(self):
        """Loads all pieces in dataset using music21.

        :param dataset_path (str): Path to dataset
        :return songs (list of m21 streams): List containing all pieces
        """
        songs = []
        count = 1

        # go through all the files in dataset and load them with music21
        for path, subdirs, files in os.walk(self.dataset_path):
            print(path, subdirs, files)
            for file in files:
                print(f"""Processing: {file} song number: {count}""")
                count += 1
                # consider only kern files
                extension = os.path.splitext(file)[1]
                if extension in self.supportedFormats:
                    try:
                        song = m21.converter.parse(os.path.join(path, file))
                    except:
                        print("Error: cannot parse ", file)
                    songs.append(song)
        return songs


    def expand_chords(self, song):
        song = m21.harmony.realizeChordSymbolDurations(song)
        #song.show('text')
        return song

    def preprocess_song(self, song):
        song = self.tr.transpose(song)
        song = self.expand_chords(song)
        return song

    def save_dataset(self, songs, pretty=False):
        with open(self.savePath, "w") as fp, open(self.savePath + "NotSaved", "w") as fe:
            saved = 0
            for song in songs:
                if song.check_properties():
                    song.write(fp, pretty)
                    fp.write("\n")
                    saved += 1
                else:
                    song.write(fe, pretty=True)
            print("Saved songs: ", saved)

    def count_chords(self, songs):
        count = 0
        for song in songs:
            song = song.flat.notesAndRests
            if song.hasElementOfClass("music21.harmony.Harmony"):
                count += 1
        return count

    def encode_song(self, song):
        """Converts a score into a time-series-like music representation. Each item in the encoded list represents 'min_duration'
        quarter lengths. The symbols used at each step are: integers for MIDI notes, 'r' for representing a rest, and '_'
        for representing notes/rests that are carried over into a new time step. Here's a sample encoding:

            ["r", "_", "60", "_", "_", "_", "72" "_"]

        :param song (m21.stream): Piece to encode
        :param time_step (float): Duration of each time step in quarter length
        :return:
        """

        encoded_song = Song(song.metadata.title)

        for event in song.flat.notesAndRests:
            if isinstance(event.duration.quarterLength, fractions.Fraction) or\
                    event.duration.quarterLength % self._time_step != 0:
                print(f"Skipping: {encoded_song.get_title()} for not supported metric")
                raise Exception('Unsupported metric')
            duration = int(event.duration.quarterLength / self._time_step)
            # handle notes
            if isinstance(event, m21.note.Note):
                symbol = event.pitch.midi  # 60
                encoded_song.add_note(symbol, duration, symbol)
            # handle rests
            elif isinstance(event, m21.note.Rest):
                symbol = "r"
                encoded_song.add_note(symbol, duration, symbol)
            elif isinstance(event, m21.harmony.ChordSymbol):
                symbol = m21.harmony.chordSymbolFigureFromChord(event)
                if symbol == "" or symbol == "Chord Symbol Cannot Be Identified":
                    symbol = self._noChordSymbol
                encoded_song.add_chord(symbol, duration, symbol)
            elif isinstance(event, m21.chord.Chord):
                # idk why in docs root return a pitch and here a string
                # fixed with a workaround
                event = m21.note.Note(event.root())
                symbol = event.pitch.midi
                encoded_song.add_note(symbol, duration, symbol)
        return encoded_song

    def preprocess_single(self, file, songs, count):
        print(f"Process {os.getpid()} processing file number: {count} called: {file} ")
        song = self.preprocess(os.path.join(self.dataset_path, file))
        if song is not None:
            songs.append(song)
        return songs

    def preprocess(self, path) -> Song:
        """Preprocess a single song"""
        extension = os.path.splitext(path)[1]
        song = None
        if extension in self.supportedFormats:
            try:
                song = m21.converter.parse(path)
                song = self.preprocess_song(song)
                song = self.encode_song(song)
            except:
                print("Error: cannot parse ", path)
                song = None
        return song



if __name__ == "__main__":
    # print(pre.count_chords(scanner.scan_multi(pre.load_songs)))
    with Chronometer() as t:
        parser = argparse.ArgumentParser(description='Process songs dataset.')
        parser.add_argument('srcPath', metavar='path', type=str,
                            help='the path of the dataset')
        parser.add_argument('destPath', metavar='destPath', type=str,
                            help='the path where save the encoded dataset')
        parser.add_argument('-pretty', action='store_true')
        args = parser.parse_args()
        pre = Preprocessor(args.srcPath, save_path=args.destPath)
        scanner = Scanner(args.srcPath)
        songs = scanner.scan_multi(pre.preprocess_single)
        print("Processed songs: ", str(len(songs)))
        print("Saving dataset...")
        pre.save_dataset(songs, pretty=args.pretty)
    print('Total time of elaboration: {:.3f} seconds'.format(float(t)))
