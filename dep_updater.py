from importlib.metadata import distributions
import subprocess
import sys

packages = [dist.metadata["Name"] for dist in distributions()]

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--upgrade", *packages],
    check=True
)
