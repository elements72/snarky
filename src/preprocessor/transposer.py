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
        srcKey = measures_part0[0][4]
        dstKey = m21.pitch.Pitch(key)

        # estimate key using music21
        if not isinstance(srcKey, m21.key.Key):
            srcKey = song.analyze("key")

        # get interval for transposition. E.g., Bmaj -> Cmaj
        if srcKey.mode == "major":
            interval = m21.interval.Interval(
                srcKey.tonic, dstKey)
        elif srcKey.mode == "minor":
            interval = m21.interval.Interval(
                srcKey.tonic, dstKey.transpose(-3))

        # transpose song by calculated interval
        tranposed_song = song.transpose(interval)
        return tranposed_song
