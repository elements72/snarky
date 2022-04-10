import music21 as m21


class Transposer:
    """_summary_
    Transpose in the given key
    """
    def __init__(self) -> None:
        pass

    def transpose(self, song, key="C"):
        """_summary_

        Args:
            song: to transpose in music21.stream.Score format
            key (str, optional): destination key. Defaults to "C".

        Returns:
            Transposed song
        """

        # get key from the song
        parts = song.getElementsByClass(m21.stream.Part)
        measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
        src_key = measures_part0[0][4]
        dst_key = m21.pitch.Pitch(key)

        # estimate key using music21
        if not isinstance(src_key, m21.key.Key):
            src_key = song.analyze("key")

        # get interval for transposition. E.g., Bmaj -> Cmaj
        if src_key.mode == "major":
            interval = m21.interval.Interval(
                src_key.tonic, dst_key)
        elif src_key.mode == "minor":
            interval = m21.interval.Interval(
                src_key.tonic, dst_key.transpose(-3))

        # transpose song by calculated interval
        tranposed_song = song.transpose(interval)
        return tranposed_song
