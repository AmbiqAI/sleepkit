"""Declare or execute a fixed saved-feature staging golden experiment."""

import argparse

from .recipe import Config, declare_golden, run
from .split import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("declare", "run"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--features-dir", required=True)
        command_parser.add_argument("--split", required=True)
        command_parser.add_argument("--output", required=True)
        if command == "declare":
            command_parser.add_argument("--name", required=True)
            command_parser.add_argument("--epochs", type=int, default=5)
            command_parser.add_argument("--batch-size", type=int, default=32)
            command_parser.add_argument("--learning-rate", type=float, default=0.001)
            command_parser.add_argument("--seed", type=int, default=0)
        else:
            command_parser.add_argument("--golden", required=True)
    args = parser.parse_args()
    if args.command == "declare":
        config = Config(
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed
        )
        declare_golden(args.features_dir, args.split, args.output, config, name=args.name)
    else:
        config = Config(**read_json(args.golden)["expected"]["config"])
        run(args.features_dir, args.split, args.output, config, golden=args.golden)
    print(args.output)


if __name__ == "__main__":
    main()
