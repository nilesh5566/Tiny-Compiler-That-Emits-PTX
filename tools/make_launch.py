"""make_launch.py - turn a test's // LAUNCH: and // ARG: directives into a launch file.

    python tools/make_launch.py tests/exec/saxpy_golden.kern > saxpy.launch
    build/bin/kernc run tests/exec/saxpy_golden.kern --launch saxpy.launch --engine ir -O1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_tests  # noqa: E402

if len(sys.argv) != 2:
    sys.exit(__doc__)
p = Path(sys.argv[1])
print(run_tests.build_launch(p.read_text(), p).text(), end="")
