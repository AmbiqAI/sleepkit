import subprocess
import sys


def test_app_cli_help():
    result = subprocess.run([sys.executable, "-m", "sleepkit", "--help"], capture_output=True, text=True, check=True)
    assert "smoke" in result.stdout
    assert "publish" in result.stdout
