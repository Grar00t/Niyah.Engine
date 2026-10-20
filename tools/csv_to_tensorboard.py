import csv, sys, math
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

csv_path = sys.argv[1] if len(sys.argv) > 1 else "train_log.csv"
run_dir = sys.argv[2] if len(sys.argv) > 2 else "runs/exp1"

w = SummaryWriter(run_dir)
n = 0
with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        step = int(row["step"])
        for key, val in row.items():
            if key == "step" or val is None or val.strip() == "":
                continue
            x = float(val)
            if math.isfinite(x):
                w.add_scalar(key, x, step)
        n += 1
w.close()
print(f"wrote {n} rows to {run_dir}")
