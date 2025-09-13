import sys
import subprocess
from importlib.resources import files

def main():
    """Console entry point: launches the packaged Streamlit app."""
    app_path = files("nlpsych_app") / "app.py"
    # Pass through any extra CLI args to Streamlit (e.g., --server.port 8502)
    args = [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    # Use check=True so non-zero exit codes surface to the shell
    subprocess.run(args, check=True)
