"""Re-serialise a saved Keras model.

Re-saving under the interpreter of the target platform avoids the cross-OS
path/encoding differences noted in the README.

    python fileconverter.py
    python fileconverter.py --source model/best_cnn_rnn_model.keras \
                            --destination LinuxEnv/best_cnn_rnn_model.keras
"""

import argparse
import os

import keras


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default=os.path.join('model', 'best_cnn_rnn_model.keras'))
    parser.add_argument('--destination',
                        default=os.path.join('LinuxEnv', 'best_cnn_rnn_model.keras'))
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source model '{args.source}' not found.")

    destination_dir = os.path.dirname(args.destination)
    if destination_dir:
        os.makedirs(destination_dir, exist_ok=True)

    model = keras.saving.load_model(args.source)
    model.save(args.destination)
    print(f"Converted {args.source} -> {args.destination}")


if __name__ == "__main__":
    main()
