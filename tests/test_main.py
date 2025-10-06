import subprocess

def test_main_runs():
    """Ensure main.py runs without error."""
    result = subprocess.run(["python", "main.py"], capture_output=True)
    assert result.returncode == 0, f"main.py failed with: {result.stderr.decode()}"