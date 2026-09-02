from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence


COMMANDS = {
    "tokens": ("exporters.tokens", "Export IMU-token motion-language pretraining data."),
    "instruction": ("exporters.instruction", "Export instruction-tuning and contrastive corpora."),
    "har": ("exporters.har", "Export the 14-dataset zero-shot HAR evaluation data."),
    "heldout": ("exporters.heldout", "Export Nymeria or EgoExo4D retrieval/captioning data."),
    "egoexo4d": ("exporters.egoexo4d", "Prepare the EgoExo4D zero-shot windows."),
    "anymo-bench": ("exporters.anymo_bench", "Export AnyMo-Bench in Hugging Face format."),
}


def _print_help() -> None:
    print("Usage: python code/export.py <command> [options]\n")
    print("Commands:")
    width = max(len(name) for name in COMMANDS)
    for name, (_, description) in COMMANDS.items():
        print(f"  {name:<{width}}  {description}")
    print("\nRun 'python code/export.py <command> --help' for command-specific options.")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0

    command = args.pop(0)
    if command not in COMMANDS:
        choices = ", ".join(COMMANDS)
        print(f"Unknown export command {command!r}. Expected one of: {choices}", file=sys.stderr)
        return 2

    module_name = COMMANDS[command][0]
    module = importlib.import_module(module_name)
    previous_argv = sys.argv
    sys.argv = [f"{previous_argv[0]} {command}", *args]
    try:
        result = module.main()
    finally:
        sys.argv = previous_argv
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())

