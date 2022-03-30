



class Song:
    def __init__(self) -> None:
        self._melody = []
        self._chords = []
        self.holdSymbol = "_"

    def get_melody(self):
        return " ".join(map(str, self._melody))

    def get_chords(self):
        return " ".join(map(str, self._chords))

    def add_symbol(self, list, symbol, duration):
        list.append(symbol)
        for _ in range(duration):
            list.append(self.holdSymbol)


    def add_note(self, note, duration):
        self.add_symbol(self._melody, note, duration)

    def add_chord(self, chord, duration):
        self.add_symbol(self._chords, chord, duration)

    def write(self, path="./dataset"):
        with open(path, "a+") as fp:
            fp.write(self.get_chords())
            fp.write("\n")
            fp.write(self.get_melody())
            fp.write("\n")

