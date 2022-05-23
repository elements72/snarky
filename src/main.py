import os.path

from net import Snarky
from loader import Loader
from preprocessor import Preprocessor
import argparse
from chronometer import Chronometer


def initialize_arguments():
    parser = argparse.ArgumentParser(description='Process songs dataset.')
    parser.add_argument('path', metavar='path', type=str,
                        help='the path of the dataset')
    parser.add_argument('-t', metavar='temperature', type=float, help='temperature value', required=False)
    parser.add_argument('-predictions', metavar='predictions', type=int, help='number of predictions', required=False)
    parser.add_argument('-source', metavar='source', type=str, help='source melody', required=False)
    parser.add_argument('-dest', metavar='dest', type=str, help='destination melody', required=False)
    parser.add_argument('-weights', metavar='params', type=str, help='params of the net', required=False)
    parser.add_argument('-weights_save', metavar='params', type=str, help='params of the net', required=False)
    parser.add_argument('-bars', help='include bars', action="store_true")
    parser.add_argument('-upbeat', help='include upbeat', action="store_true")
    parser.add_argument('-time_step', help='Time step', type=float)

    return parser.parse_args()


if __name__ == "__main__":
    with Chronometer() as t:
        args = initialize_arguments()
        path = args.path
        source_melody = args.source

        batch_size = 64
        sequence_length = 32
        buffer_size = batch_size - sequence_length

        params = ["chords", "chords_play", "melody", "melody_play"]
        if args.bars:
            params.append("bars")
        if args.upbeat:
            params.append("upbeat")

        num_predictions = args.predictions if args.predictions else 100
        dest_path = args.dest if args.dest else "./generated"
        weights = args.weights if args.weights else None
        weights_save = args.weights_save if args.weights_save else None

        # Load the dataset and create the sequence
        loader = Loader(os.path.join(path, "train"), _params=params)
        dataset = loader.load(categorical=True)
        sequence = loader.create_sequences(dataset, sequence_length)
        params = loader.get_params()

        # Create the modelinputs
        snarky = Snarky(_sequence=sequence, _batch_size=batch_size, _sequence_length=sequence_length, _params=params)

        # snarky.create_model(num_units=512)
        snarky.autoencoder(num_units=128)

        if weights is not None:
            snarky.load(weights)

        snarky.summary()
        snarky.train()


        if source_melody is not None:
            # Load a song for generation input
            pre = Preprocessor("./", params, time_step=args.time_step)
            song = pre.preprocess(source_melody)
            encoded_song = loader.encode_song(song.get_song(bars=args.bars))
            generated = snarky.generate(encoded_song, num_predictions=num_predictions)


        loader.save_song(generated, dest_path)
        print(generated)


