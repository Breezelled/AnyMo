from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence


COMMANDS = {
    "candidates": (
        "simulation.surface_candidates",
        "Select body-surface candidate placements for the 23 anatomical segments.",
    ),
    "frames": (
        "simulation.construct_local_sensor_frames",
        "Construct tangent/binormal/normal local sensor frames.",
    ),
    "generate": (
        "simulation.simulate_geometry_aware_imu",
        "Generate geometry-aware body-surface IMU signals with WIMUSim.",
    ),
    "convert": (
        "simulation.storage",
        "Convert simulated IMU arrays to time-chunked Zarr storage.",
    ),
}


def _print_help() -> None:
    print("Usage: python code/simulate.py <command> [options]\n")
    print("Commands:")
    width = max(len(name) for name in COMMANDS)
    for name, (_, description) in COMMANDS.items():
        print(f"  {name:<{width}}  {description}")
    print("\nRun 'python code/simulate.py <command> --help' for command-specific options.")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0

    command = args.pop(0)
    if command not in COMMANDS:
        choices = ", ".join(COMMANDS)
        print(f"Unknown simulation command {command!r}. Expected one of: {choices}", file=sys.stderr)
        return 2

    module = importlib.import_module(COMMANDS[command][0])
    previous_argv = sys.argv
    sys.argv = [f"{previous_argv[0]} {command}", *args]
    try:
        result = module.main()
    finally:
        sys.argv = previous_argv
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
