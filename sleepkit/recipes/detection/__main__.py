"""Additive commands for one concrete recipe, not a universal experiment runner."""

import argparse
import json
from pathlib import Path
import tempfile

from .recipe import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--split", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--cache", type=Path)
    train.add_argument("--context", type=int, default=240)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=0)
    inference = commands.add_parser("predict")
    inference.add_argument("--bundle", required=True, type=Path)
    inference.add_argument("--data", required=True, type=Path)
    inference.add_argument("--subject", required=True)
    inference.add_argument("--output", required=True, type=Path)
    smoke = commands.add_parser("smoke", help="End-to-end synthetic smoke run; no accuracy claim")
    smoke.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "train":
        from .recipe import run

        output = run(
            args.data,
            args.split,
            args.output,
            Config(args.context, args.epochs, args.batch_size, args.learning_rate, args.seed),
            cache=args.cache,
        )
        print(json.dumps({"run": str(output)}))
    elif args.command == "predict":
        import numpy as np
        from .data import read_subject
        from .inference import predict

        data, _ = read_subject(args.data, args.subject, labels=False)
        result = predict(args.bundle, data)
        with args.output.open("xb") as stream:
            np.savez(stream, **result)
    else:
        from .smoke import run_smoke

        with tempfile.TemporaryDirectory(prefix="sleepkit-synthetic-") as directory:
            run_smoke(Path(directory), args.output)


if __name__ == "__main__":
    main()
