import os.path

import numpy as np

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
        sequence_length = 32
        buffer_size = batch_size - sequence_length

        num_predictions = args.predictions if args.predictions else 125
        dest_path = args.dest if args.dest else "./generated"
        weights = args.weights if args.weights else None
        weights_save = args.weights_save if args.weights_save else None

        # Load the dataset and create the sequence
        loader = Loader(os.path.join(path, "train"))
        dataset = loader.load()
        sequence = loader.create_sequences2(dataset, sequence_length)
        params = loader.get_params()

        """# Load the test dataset
        loader_test = Loader(os.path.join(path, "test"))
        dataset_test = loader_test.load()
        sequence_test = loader_test.create_sequences(dataset_test, sequence_length)
        params_test = loader_test.get_params()
        #sequence_test = (sequence_test.shuffle(buffer_size).batch(batch_size, drop_remainder=True).cache())

        # Load a single track and create sequence
        loader_single = Loader(path="../../datasets/single/train")
        dataset_single = loader_single.load(loader.get_vocabulary())
        sequence_single = loader_single.create_sequences(dataset_single, sequence_length)
        params_single = loader_single.get_params()
        """

        # Create the model
        snarky = Snarky(_sequence=sequence, _batch_size=batch_size, _sequence_length=sequence_length, _params=params)
        snarky.create_model2()

        # Load a song for generation input
        pre = Preprocessor("./")
        song = pre.preprocess(source_melody)
        encoded_song = loader.encode_song(song.get_song())
        #if weights is not None:
        #    snarky.load(weights)

        # print(sequence.element_spec)
        # print(sequence_test.element_spec)
        #snarky.train(epochs=1)

        #snarky.evaluate(sequence.batch(batch_size))

        #if weights_save is not None:
        #    snarky.save(weights_save)



        n = 1000
        correct = {"chords": 0, "chords_play": 0, "melody": 0, "m elody_play": 0}
        """for seq, target in sequence_single.take(n):
           # print("Input seq:", seq)
            generated = snarky.generate(seq, num_predictions=1)
            generated = generated.squeeze(axis=0)
            y = []
            for targ, gen in zip(target, generated):
                # print(target, target[targ])
                y.append(target[targ].numpy())
                if target[targ] == gen:
                    correct[targ] += 1
            print(f"Target: {y}, output{generated}")
        print(f"Accuracy= {[(correct[key]/n)*100 for key in correct]}")

        # print("Inputs2: ", encoded_song[:sequence_length])"""
        generated = snarky.generate(encoded_song, num_predictions=n)
        #for target, gen in zip(encoded_song[sequence_length:], generated):
        #    print(f"Target: {target}, generated{gen}")
        loader.save_song(generated, dest_path)
        print(generated)


