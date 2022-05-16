import os.path
import re
from net import Snarky
from loader import Loader
from preprocessor import Preprocessor
import argparse
from decoder import Decoder
import pathlib
from chronometer import Chronometer


def initialize_arguments():
    parser = argparse.ArgumentParser(description='Process songs dataset.')
    parser.add_argument('path', metavar='path', type=str, help='the path of the data')
    parser.add_argument('-predictions', metavar='predictions', type=int, help='number of predictions', required=True)
    parser.add_argument('-source', metavar='source', type=str, help='source melody', required=True)

    return parser.parse_args()



def generate(dir, source_melody, bars=False, time_step=0.125 ,num_predictions=100, dest="generated", weights="model.params"):
    batch_size = 64
    sequence_length = 32
    buffer_size = batch_size - sequence_length

    num_predictions = int(num_predictions * (4/time_step))
    params = ["chords", "chords_play", "melody", "melody_play"]
    if bars:
        params.append("bars")

    dest = pathlib.PurePath(source_melody).stem
    dest_path = os.path.join(dir, dest)
    weights = os.path.join(dir, weights)

    # Load the dataset and create the sequence
    loader = Loader(os.path.join(dir, "train"), _params=params)
    dataset = loader.load()
    sequence = loader.create_sequences2(dataset, sequence_length)
    params = loader.get_params()

    # Decoder
    decoder = Decoder(time_step=time_step)

    # Create the model
    snarky = Snarky(_sequence=sequence, _batch_size=batch_size, _sequence_length=sequence_length, _params=params)
    snarky.create_model2()

    # Load a song for generation input
    pre = Preprocessor("./", params, time_step=time_step)
    song = pre.preprocess(source_melody)
    encoded_song = loader.encode_song(song.get_song(bars=bars))

    snarky.load(weights)
    generated = snarky.generate(encoded_song, num_predictions=num_predictions)

    loader.save_song(generated, dest_path)
    decoder.create_midi(dest_path)
    return generated


if __name__ == "__main__":
    args = initialize_arguments()
    path = args.path
    for dir in os.listdir(path):
        if "B" in dir:
            bars = True
        else:
            bars = False
        numbers = re.findall('[0-9]+', dir)
        if len(numbers) > 0:
            time_step = 4 / int(numbers[0])
        else:
            continue
        try:
            generate(os.path.join(path, dir), bars=bars, source_melody=args.source, num_predictions=args.predictions,
                     time_step=time_step)
        except:
            print(f"Failed to generate: {dir}")
