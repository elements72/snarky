import os.path
import re
from net import Snarky
from loader import Loader
from preprocessor import Preprocessor
import argparse
from decoder import Decoder
import pathlib
import glob
from chronometer import Chronometer


def initialize_arguments():
    parser = argparse.ArgumentParser(description='Process songs dataset.')
    parser.add_argument('path', metavar='path', type=str, help='the path of the data')
    parser.add_argument('-predictions', metavar='predictions', type=int, help='number of predictions', required=True)
    parser.add_argument('-source', metavar='source', type=str, help='source melody', required=True)

    return parser.parse_args()



def generate(dir, source_melody, bars=False, upbeat=False, time=32 ,num_predictions=100, dest="generated", weights="model.params", num_units=128):
    batch_size = 64
    sequence_length = 32
    buffer_size = batch_size - sequence_length
    time_step = 4 / int(time)

    num_predictions = int(num_predictions * (4/time_step))
    params = ["chords", "chords_play", "melody", "melody_play"]
    if bars:
        params.append("bars")
    if upbeat:
        params.append("upbeat")

    dest = pathlib.PurePath(source_melody).stem
    dest_path = os.path.join(dir, dest)
    weights = os.path.join(dir, weights)

    # Load the dataset and create the sequence
    loader = Loader(os.path.join(dir, "train"), _params=params)
    dataset = loader.load()
    sequence = loader.create_sequences(dataset, sequence_length)
    params = loader.get_params()


    for weights in glob.glob(f"{weights}*.index"):
        autoencoder = "A" in weights
        save_path = dest_path
        weights = pathlib.PurePath(weights)
        num_units = re.findall('[0-9]+', weights.name)
        if len(num_units) == 0:
            num_units = 128
        else:
            num_units = int(num_units[0])
        print(f"Processing dir: {dir}, with num_units: {num_units}")
        name = f"{dest}_{time}{'B' if bars else ''}_{'U_' if upbeat else ''}{'A_' if autoencoder else ''}{num_units}"
        save_path = f"{save_path}_{time}{'B' if bars else ''}_{num_units}"
        # Create the model
        snarky = Snarky(_sequence=sequence, _batch_size=batch_size, _sequence_length=sequence_length, _params=params)
        if autoencoder:
            snarky.autoencoder(num_units=num_units)
        else:
            snarky.create_model(num_units=num_units)

        # Load a song for generation input
        pre = Preprocessor("./", params, time_step=time_step)
        song = pre.preprocess(source_melody)
        encoded_song = loader.encode_song(song.get_song(bars=bars, upbeat=upbeat))

        snarky.load(os.path.splitext(weights.as_posix())[0])
        generated = snarky.generate(encoded_song, num_predictions=num_predictions)

        loader.save_song(generated, save_path)
        # Decoder
        decoder = Decoder(time_step=time_step)

        decoder.show_midi(path=save_path, name=name)
    return generated


if __name__ == "__main__":
    args = initialize_arguments()
    path = args.path
    for dir in os.listdir(path):
        if dir != 'dataSM8B':
            continue
        if "B" in dir:
            bars = True
        else:
            bars = False
        if "U" in dir:
            upbeat = True
        else:
            upbeat = False
        numbers = re.findall('[0-9]+', dir)
        if len(numbers) > 0:
            time = numbers[0]
        else:
            continue
        try:
            generate(os.path.join(path, dir), bars=bars, source_melody=args.source, num_predictions=args.predictions,
                     time=time, weights="model.params", upbeat=upbeat)
        except Exception as e:
            print(f"Failed to generate: {dir}, error: {e}")
