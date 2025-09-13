import sys
from pathlib import Path
import sys, os

print("CWD:", os.getcwd())
print("sys.path head:", sys.path[:5])

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlpsych_app.app import main  # now safe
if __name__ == "__main__":
    main()