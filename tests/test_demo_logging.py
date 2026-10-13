"""Native demo labels and exclusive output behavior; no model training."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

exe = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix="niyah-synthetic-demo-") as name:
    root = Path(name)
    result = subprocess.run([str(exe)], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "SYNTHETIC DEMO" in result.stderr
    path = root / "synthetic_demo_log.csv"
    raw = path.read_bytes()
    assert raw.startswith(b"synthetic_step,synthetic_loss,")
    assert not (root / "train_log.csv").exists()
    before = hashlib.sha256(raw).hexdigest()
    second = subprocess.run([str(exe)], cwd=root, capture_output=True, text=True)
    assert second.returncode != 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    print("PASS synthetic_labels exclusive_creation existing_bytes_preserved")
