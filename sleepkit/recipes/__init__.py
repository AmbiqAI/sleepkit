"""Thin CLI around ordinary experiment and deployment functions."""


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="SleepKit 1.0 development recipes")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="Run the synthetic raw-record staging example")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--epochs", type=int, default=2)
    train = commands.add_parser("train", help="Train the reference model on portable raw records")
    train.add_argument("--records", nargs="+", required=True)
    train.add_argument("--preprocessing", required=True, help="WindowFeatures JSON definition")
    train.add_argument("--class-names", nargs="+", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=2)
    evaluate = commands.add_parser("evaluate", help="Evaluate a deployed bundle on held-out raw records")
    evaluate.add_argument("--bundle", required=True)
    evaluate.add_argument("--records", nargs="+", required=True)
    validate = commands.add_parser("validate", help="Validate a local deployment bundle")
    validate.add_argument("path")
    publish = commands.add_parser("publish", help="Stage a bundle for Hugging Face; dry-run by default")
    publish.add_argument("path")
    publish.add_argument("repo_id")
    publish.add_argument("--upload", action="store_true")
    publish.add_argument("--public", action="store_true")
    publish.add_argument("--license-file")
    args = parser.parse_args()
    if args.command in {"smoke", "train"}:
        if args.epochs < 1:
            parser.error("epochs must be positive")
        from .staging import run

        kwargs = {}
        if args.command == "train":
            from pathlib import Path
            from sleepkit.data.readers import read_record
            from sleepkit.preprocessing import WindowFeatures

            kwargs = {
                "records": [read_record(path) for path in args.records],
                "preprocessing": WindowFeatures.from_dict(json.loads(Path(args.preprocessing).read_text())),
                "class_names": args.class_names,
            }
        print(json.dumps(run(args.output, epochs=args.epochs, **kwargs), indent=2))
    elif args.command == "evaluate":
        from sleepkit.data.readers import read_record
        from sleepkit.evaluation import evaluate_records
        from sleepkit.runtime import Predictor

        print(
            json.dumps(evaluate_records(Predictor(args.bundle), [read_record(path) for path in args.records]), indent=2)
        )
    elif args.command == "validate":
        from sleepkit.runtime import Predictor

        predictor = Predictor(args.path)
        predictor.check_reference()
        print("Bundle checksums, tensor contract, and reference inference passed")
    else:
        from sleepkit.export import publish_bundle

        print(
            publish_bundle(
                args.path,
                args.repo_id,
                dry_run=not args.upload,
                private=not args.public,
                license_file=args.license_file,
            )
        )
