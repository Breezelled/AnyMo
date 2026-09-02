from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence


COMMANDS = {
    "imu": ("nymeria_sync.imu", "Synchronize the six head/wrist IMU streams to 60 Hz."),
    "motion": ("nymeria_sync.motion", "Synchronize Xsens motion and atomic-action annotations."),
    "mesh": ("nymeria_sync.mesh", "Synchronize the Momentum body mesh to the 60 Hz grid."),
    "narration": ("nymeria_sync.narration", "Synchronize motion narrations to the 60 Hz grid."),
}


def _print_help() -> None:
    print("Usage: python code/sync_nymeria.py <command> [options]\n")
    print("Commands:")
    width = max(len(name) for name in COMMANDS)
    for name, (_, description) in COMMANDS.items():
        print(f"  {name:<{width}}  {description}")
    print("\nAll commands require the official Nymeria tools.")
    print("Run 'python code/sync_nymeria.py <command> --help' for command-specific options.")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0

    command = args.pop(0)
    if command not in COMMANDS:
        choices = ", ".join(COMMANDS)
        print(f"Unknown synchronization command {command!r}. Expected one of: {choices}", file=sys.stderr)
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

