
class Song:
    def __init__(self, title, no_chord_symbol="NC", time_step=0.125) -> None:
        self._title = title
        self._melody: list[str] = []
        self._chords: list[str] = []
        self._melody_holder: list[str] = []
        self._chords_holder: list[str] = []
        self._bars: list[str] = []
        self._noChordSymbol = no_chord_symbol
        self._tempo = 4
        self._time_step = int(self._tempo / time_step)
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

    def get_song(self, bars=False):
        self.pad_chords()
        song = {"chords": self._chords, "chords_play": self._chords_holder, "melody": self._melody,
                "melody_play": self._melody_holder}
        if bars:
            self.get_bars()
            song["bars"] = self._bars
        return song

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

    def split_bar(self, split: int):
        bars = []
        for i in range(self._songDuration):
            if i % split == 0:
                bars.append("1")
            else:
                bars.append("0")
        return bars

    def get_bars(self):
        nbars = int(self._songDuration / self._time_step)
        bars_len = nbars + self._songDuration + 1
        if len(self._bars) < bars_len:
            self._bars = self.split_bar(self._time_step)
        return " ".join(map(str, self._bars))

    def get_upbeat(self):
        split = int(self._time_step / 8)
        upbeat = []
        symbol = "0"
        for i in range(self._songDuration):
            if i % split == 0:
                symbol = "1" if symbol == "0" else "0"
            upbeat.append(symbol)
        return " ".join(map(str, upbeat))



    def get_title(self):
        return self._title

    def check_properties(self):
        """
        The song must have the same number of chords and notes, if not is in an illegal format
        :return:
        :rtype:
        """
        # Pad the chords with NC symbol at the start
        self.pad_chords()
        return len(self._chords) == len(self._melody)

    def write(self, fp, pretty=False, bars=False, upbeat=False):
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
        if bars:
            fp.write("\n")
            fp.write(self.get_bars())
        if upbeat:
            fp.write("\n")
            fp.write(self.get_upbeat())
        if pretty:
            fp.write("\n")
            fp.write(f"{str(len(self._chords))} ")
            fp.write(str(len(self._melody)))


