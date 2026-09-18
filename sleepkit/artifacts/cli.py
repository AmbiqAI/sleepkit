"""Additive artifact CLI; the existing SleepKit training CLI is unchanged."""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage-baseline", help="Stage the reviewed historical SD-2-TCN-SM baseline")
    stage.add_argument("--source", required=True)
    stage.add_argument("--output", required=True)
    stage.add_argument("--baseline", default="sd-2-tcn-sm")
    stage.add_argument("--runtime-check", action="store_true")
    stage.add_argument("--license-file")
    stage.add_argument("--license-id")
    validate = commands.add_parser("validate", help="Verify integrity and optionally replay synthetic I/O")
    validate.add_argument("path")
    validate.add_argument("--profile", choices=("archive", "runnable"), default="archive")
    validate.add_argument("--runtime", action="store_true")
    publish = commands.add_parser("publish", help="Preview publication; --upload performs the remote write")
    publish.add_argument("path")
    publish.add_argument("repo_id")
    publish.add_argument("--profile", choices=("archive", "runnable"), default="archive")
    publish.add_argument("--upload", action="store_true")
    publish.add_argument(
        "--public",
        action="store_true",
        help="Create a public repository (default: private); existing visibility is unchanged",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "stage-baseline":
            from .baselines import stage_baseline

            result = {
                "path": str(
                    stage_baseline(
                        args.source,
                        args.output,
                        baseline=args.baseline,
                        runtime_check=args.runtime_check,
                        license_file=args.license_file,
                        license_id=args.license_id,
                    )
                )
            }
        elif args.command == "validate":
            from .package import validate_bundle

            result = validate_bundle(args.path, profile=args.profile)
            if args.runtime:
                from .runtime import replay_bundle

                result["runtime_replay"] = replay_bundle(args.path)
        else:
            from .hub import publish_bundle

            result = publish_bundle(
                args.path, args.repo_id, upload=args.upload, private=not args.public, profile=args.profile
            )
    except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(result, indent=2))
