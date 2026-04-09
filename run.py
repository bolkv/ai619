"""CLI wrapper around ``logic.run`` for the plugin submission format.

Reads a JSON payload from ``--input``, invokes ``logic.run``, and writes the
result as JSON to ``--output``.
"""

import argparse
import json
from pathlib import Path

from logic import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))

    result = run(payload)

    Path(args.output).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
