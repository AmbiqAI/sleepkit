"""Run the synthetic signal recipe with a small explicit configuration."""

import argparse
import json
from pathlib import Path

from .recipe import Config, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    output = run(args.output, Config(args.batch_size, args.epochs, args.seed))
    print(json.dumps({"run": str(output)}))


if __name__ == "__main__":
    main()
