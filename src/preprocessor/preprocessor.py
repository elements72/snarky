import os
import music21 as m21
import transposer as tr
from song import Song

class Preprocessor:
    """_summary_
    Preprocess data 
    """
    def __init__(self, datasets_paths) -> None:
        self.datasets_paths = datasets_paths
        self.supportedFormats = [".mid", ".krn", ".mxl"]
        self.tr = tr.Transposer()

    def load_songs(self):
        """Loads all kern pieces in dataset using music21.

        :param dataset_path (str): Path to dataset
        :return songs (list of m21 streams): List containing all pieces
        """
        songs = []

        # go through all the files in dataset and load them with music21
        for dataset_path in self.datasets_paths:
            print(dataset_path)
            for path, subdirs, files in os.walk(dataset_path):
                print(path, subdirs, files)
                for file in files:
                    print("Processing: " + file)
                    # consider only kern files
                    extension = os.path.splitext(file)[1]
                    if extension in self.supportedFormats:
                        song = m21.converter.parse(os.path.join(path, file))
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


    def encode_song(self, song, time_step=0.25):
        """Converts a score into a time-series-like music representation. Each item in the encoded list represents 'min_duration'
        quarter lengths. The symbols used at each step are: integers for MIDI notes, 'r' for representing a rest, and '_'
        for representing notes/rests that are carried over into a new time step. Here's a sample encoding:

            ["r", "_", "60", "_", "_", "_", "72" "_"]

        :param song (m21 stream): Piece to encode
        :param time_step (float): Duration of each time step in quarter length
        :return:
        """

        encoded_song = Song()

        nChord = 0
        for event in song.flat.notesAndRests:
            duration = int(event.duration.quarterLength / time_step)
            print("Event: ", event)
            print("Duration: ", duration)
            # handle notes
            if isinstance(event, m21.note.Note):
                symbol = event.pitch.midi  # 60
                encoded_song.add_note(symbol, duration)
            # handle rests
            elif isinstance(event, m21.note.Rest):
                symbol = "r"
                encoded_song.add_note(symbol, duration)
            elif isinstance(event, m21.harmony.ChordSymbol):
                symbol = m21.harmony.chordSymbolFigureFromChord(event)
                encoded_song.add_chord(symbol, duration)
                nChord += 1
            else:
                print(event)

        print(nChord)
        return encoded_song

    def preprocess(self):
        print("Loading songs...")
        songs = self.load_songs()
        for i, song in enumerate(songs):
            songs[i] = self.preprocess_song(song)
        return songs
        


if __name__ == "__main__":
    pre = Preprocessor(["../../dataset"])
    songs = pre.preprocess()
    for song in songs:
        song = pre.encode_song(song)
        print("Chords: ",len(song._chords))
        print("Melody: ",len(song._melody))
        song.write()