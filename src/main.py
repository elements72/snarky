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
    parser.add_argument('-autoencoder', help='use autoencoder', action="store_true")
    parser.add_argument('-evaluate', help='evaluate the net', action="store_true")
    parser.add_argument('-time_step', help='Time step', type=float)
    parser.add_argument('-temperature', help='temperature', type=float)
    parser.add_argument('-num_units', help='num units', type=int)

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

        tempo = 4
        num_predictions = args.predictions if args.predictions else 12
        num_predictions = int(num_predictions * (4 / args.time_step))
        dest_path = args.dest if args.dest else "./generated"
        weights = args.weights if args.weights else None
        num_units = args.num_units if args.num_units else 128
        weights_save = args.weights_save if args.weights_save else None
        time_division = int(tempo / args.time_step)

        # Load the dataset and create the sequence
        loader = Loader(os.path.join(path, "train"), _params=params)
        dataset = loader.load(categorical=True)
        sequence = loader.create_sequences(dataset, sequence_length, latent=args.autoencoder)
        params = loader.get_params()

        # Create the model inputs
        if args.autoencoder:
            snarky = Snarky(_sequence=sequence, _batch_size=1, _sequence_length=sequence_length, _params=params)
            snarky.vae(num_units=num_units, time_step=time_division)
        else:
            snarky = Snarky(_sequence=sequence, _batch_size=batch_size, _sequence_length=sequence_length, _params=params)
            snarky.create_model(num_units=num_units)

        snarky.summary()
        snarky.plot_model()
        if weights is not None:
            snarky.load(weights)

        #snarky.vae(num_units=num_units, time_step=time_division)
        if args.evaluate:

            # Load the dataset and create the sequence for test
            loader_test = Loader(os.path.join(path, "test"), _params=params)
            dataset_test = loader_test.load(vocab=loader.get_vocabulary(), categorical=True)
            sequence_test = loader_test.create_sequences(dataset_test, sequence_length, latent=args.autoencoder)

            print("Evaluating on train...")
            train_results = snarky.evaluate(sequence.shuffle(buffer_size).batch(batch_size))
            print("Evaluatin on test...")
            test_results = snarky.evaluate(sequence_test.shuffle(buffer_size).batch(batch_size))
            import json

            with open(os.path.join(path, f"{num_units}_metrics"), "w") as fp:
                json.dumps(train_results, fp)
                json.dumps(test_results, fp)


        if source_melody is not None:
            print("Generating...")
            # Load a song for generation input
            pre = Preprocessor("./", params, time_step=args.time_step)
            song = pre.preprocess(source_melody)
            encoded_song = loader.encode_song(song.get_song(bars=args.bars, upbeat=args.upbeat))
            temperature = args.temperature if args.temperature else 1
            generated = snarky.generate(encoded_song, num_predictions=num_predictions, temperature=temperature, latent=args.autoencoder)
            loader.save_song(generated, dest_path)
            print(generated)




