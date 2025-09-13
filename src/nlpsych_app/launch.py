import os, sys, subprocess
from importlib.resources import files
from importlib.util import find_spec

def main():
    # Check dependency from the [app] extra
    if find_spec("streamlit") is None:
        print(
            "Streamlit is not installed.\n\n"
            "Install the app extra and try again:\n"
            "  pip install 'NLPsych[app]'\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Respect user config if set; otherwise point to packaged config
    if "STREAMLIT_CONFIG_DIR" not in os.environ:
        cfg_dir = files("nlpsych_app") / ".streamlit"
        os.environ["STREAMLIT_CONFIG_DIR"] = str(cfg_dir)

    app_path = files("nlpsych_app") / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]], check=True)
