from net import Snarky
from loader import Loader
from preprocessor import Preprocessor, Song
import argparse
from chronometer import Chronometer


def initialize_arguments():
    parser = argparse.ArgumentParser(description='Process songs dataset.')
    parser.add_argument('path', metavar='path', type=str,
                        help='the path of the dataset')
    parser.add_argument('-t', metavar='temperature', type=float, help='temperature value', required=False)
    parser.add_argument('-predictions', metavar='temperature', type=int, help='temperature value', required=False)
    parser.add_argument('-source', metavar='source', type=str, help='source melody', required=False)
    parser.add_argument('-dest', metavar='dest', type=str, help='destination melody', required=False)

    return parser.parse_args()


if __name__ == "__main__":
    with Chronometer() as t:
        args = initialize_arguments()
        path = args.path
        source_melody = args.source

        batch_size = 64
        sequence_length = 25
        num_predictions = args.predictions if args.predictions else 125
        dest_path = args.dest if args.dest else "./generated"

        # Load the dataset and create the sequence
        loader = Loader(path)
        dataset = loader.load()
        sequence = loader.create_sequences(dataset, sequence_length)
        params = loader.get_params()

        # Create the model
        snarky = Snarky(_sequence=sequence, _batch_size=batch_size, _sequence_length=sequence_length, _params=params)
        snarky.create_model()

        # Load a song for generation input
        pre = Preprocessor("./")
        song = pre.preprocess(source_melody)
        encoded_song = loader.encode_song(song.get_song())

        generated = snarky.generate(encoded_song[:sequence_length], num_predictions=384)

        loader.save_song(generated, dest_path)
