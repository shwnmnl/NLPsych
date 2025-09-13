import os, sys, subprocess
from importlib.resources import files

def main():
    # Respect user config if set; otherwise point to packaged config
    if "STREAMLIT_CONFIG_DIR" not in os.environ:
        cfg_dir = files("nlpsych_app") / ".streamlit"
        os.environ["STREAMLIT_CONFIG_DIR"] = str(cfg_dir)

    app_path = files("nlpsych_app") / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]],
        check=True,
    )