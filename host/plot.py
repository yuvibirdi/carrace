"""Draw the loss curves of the last training run: .venv/bin/python host/plot.py [data-dir]"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = sys.argv[1] if len(sys.argv) > 1 else "data"
rows = list(csv.DictReader(open(f"{folder}/train_log.csv")))
epochs = [int(r["epoch"]) for r in rows]
held = [float(r["held_out_loss"]) for r in rows]
plt.figure(figsize=(7, 4))
plt.plot(epochs, [float(r["train_loss"]) for r in rows], label="training episodes")
plt.plot(epochs, held, label="held-out episodes")
best = held.index(min(held))
plt.scatter([epochs[best]], [held[best]], color="black", zorder=3, label=f"saved network (epoch {epochs[best]})")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(0)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{folder}/train_loss.png", dpi=130)
print(f"{folder}/train_loss.png")
