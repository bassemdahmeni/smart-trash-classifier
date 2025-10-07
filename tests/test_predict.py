import subprocess

def test_predict_runs():
    """Ensure predict.py runs successfully."""
    result = subprocess.run(
        ["python", "research\\01_data_ingestion.ipynb"], capture_output=True
    )
    assert result.returncode == 0, f"predict.py failed with: {result.stderr.decode()}"