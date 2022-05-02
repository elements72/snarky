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
    parser.add_argument('-predictions', metavar='temperature', type=int, help='temperature value', required=False)
    parser.add_argument('-source', metavar='source', type=str, help='source melody', required=False)
    parser.add_argument('-dest', metavar='dest', type=str, help='destination melody', required=False)
    parser.add_argument('-weights', metavar='params', type=str, help='params of the net', required=False)
    parser.add_argument('-weights_save', metavar='params', type=str, help='params of the net', required=False)

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
        weights = args.weights if args.weights else None
        weights_save = args.weights_save if args.weights_save else None

        # Load the dataset and create the sequence
        loader = Loader(path)
        dataset = loader.load()
        sequence = loader.create_sequences(dataset, sequence_length)
        params = loader.get_params()

        loader1 = Loader(path="../../datasets/dataMD")
        dataset1 = loader.load()
        sequence1 = loader.create_sequences(dataset1, sequence_length)
        params1 = loader.get_params()

        # Create the model
        snarky = Snarky(_sequence=sequence, _batch_size=batch_size, _sequence_length=sequence_length, _params=params)
        snarky.create_model()

        # Load a song for generation input
        pre = Preprocessor("./")
        song = pre.preprocess(source_melody)
        encoded_song = loader.encode_song(song.get_song())

        if weights is not None:
            snarky.load(weights)

        snarky.train(epochs=1)

        if weights_save is not None:
            snarky.save(weights_save)


        for seq, target in sequence1.take(10):
            generated = snarky.generate(seq, num_predictions=1)
            y = []
            for x in target:
                y.append(target[x].numpy())
            print(f"Target: {y}, output{generated}")
        print("---------------")
        for seq, target in sequence.take(10):
            generated = snarky.generate(seq, num_predictions=1)
            y = []
            for x in target:
                y.append(target[x].numpy())
            print(f"Target: {y}, output{generated}")

        loader.save_song(generated, dest_path)


