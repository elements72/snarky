



class Song:
    def __init__(self, hold_symbol="_", no_chord_symbol="NC") -> None:
        self._melody = []
        self._chords = []
        self._holdSymbol = hold_symbol
        self._noChordSymbol = no_chord_symbol
        self._songDuration = 0

    def get_melody(self):
        return " ".join(map(str, self._melody))

    def get_chords(self):
        if len(self._chords) == 0:
            self.add_chord(self._noChordSymbol, self._songDuration)
        return " ".join(map(str, self._chords))

    def add_symbol(self, list, symbol, duration):
        list.append(symbol)
        for _ in range(duration):

            list.append(self._holdSymbol)


    def add_note(self, note, duration):
        self._songDuration += 1
        self.add_symbol(self._melody, note, duration)

    def add_chord(self, chord, duration):
        self.add_symbol(self._chords, chord, duration)

    def write(self, fp):
        fp.write(self.get_chords())
        fp.write("\n")
        fp.write(self.get_melody())
