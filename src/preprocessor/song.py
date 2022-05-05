
class Song:
    def __init__(self, title, no_chord_symbol="NC") -> None:
        self._title = title
        self._melody: list[str] = []
        self._chords: list[str] = []
        self._melody_holder: list[str] = []
        self._chords_holder: list[str] = []
        self._noChordSymbol = no_chord_symbol
        self._songDuration = 0

    def get_melody(self):
        return " ".join(map(str, self._melody))

    def get_melody_holder(self):
        return " ".join(map(str, self._melody_holder))

    def get_chords_holder(self):
        if len(self._chords) == 0:
            self.add_chord(self._noChordSymbol, self._songDuration, self._noChordSymbol)
        return " ".join(map(str, self._chords_holder))

    def get_chords(self):
        self.pad_chords()
        return " ".join(map(str, self._chords))

    def get_song(self, params=["chords", "chords_play", "melody", "melody_play"]):
        self.pad_chords()
        return {"chords": self._chords, "chords_play": self._chords_holder, "melody": self._melody,
                "melody_play": self._melody_holder}

    def pad_chords(self):
        if len(self._chords) == 0:
            self.add_chord(self._noChordSymbol, self._songDuration, self._noChordSymbol)
        if len(self._chords) < self._songDuration:
            self.add_chord(self._noChordSymbol, self._songDuration - len(self._chords),
                           self._noChordSymbol, append=False)

    def add_symbol(self, list, holder, symbol, duration, hold_symbol="_", append=True):
        if duration > 0:
            position = len(list) if append else 0
            list.insert(position, str(symbol))
            holder.insert(position, "1")
            for i in range(1, duration):
                list.insert(position + i, str(hold_symbol))
                holder.insert(position + i, "0")

    def add_note(self, note, duration, hold_symbol="_"):
        self._songDuration += duration
        self.add_symbol(self._melody, self._melody_holder, note, duration, hold_symbol)

    def add_chord(self, chord, duration, hold_symbol="_", append=True):
        self.add_symbol(self._chords, self._chords_holder, chord, duration, hold_symbol, append)

    def get_title(self):
        return self._title

    def check_properties(self):
        self.pad_chords()
        return len(self._chords) == len(self._melody)

    def write(self, fp, pretty=False):
        if pretty:
            fp.write(self._title)
            fp.write("\n")
        fp.write(self.get_chords())
        fp.write("\n")
        fp.write(self.get_chords_holder())
        fp.write("\n")
        fp.write(self.get_melody())
        fp.write("\n")
        fp.write(self.get_melody_holder())
        if pretty:
            fp.write("\n")
            fp.write(f"{str(len(self._chords))} ")
            fp.write(str(len(self._melody)))


