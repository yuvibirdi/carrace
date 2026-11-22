"""Train the driving network on the recorded episodes (behaviour cloning).

    .venv/bin/python host/episodes.py          # fetch new episodes from the Pi into data/
    .venv/bin/python host/train.py [epochs] [data-dir]

Every frame is a training example: the network sees the frame and has to predict
the actions the driver had applied at that frame and the CHUNK - 1 frames after it.
Every 5th episode is held out whole, to measure how well it does on laps it never saw.
The best network by held-out loss is saved to <data-dir>/model.pt, and the numbers
for every epoch go to <data-dir>/train_log.csv.
"""
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

import episodes
from model import CHUNK, LEFT, RIGHT, STRAIGHT, Net, prepare, steer_class

FRAME_MS = 1000 / 30
BATCH = 256


def load(folder):
    """Returns frames [N, H, W, 3], per-frame steer class and throttle flag, the indices
    usable as training examples split into train and held-out, and the usual drive speed."""
    frames, steer, throttle, train, held, speeds = [], [], [], [], [], []
    kept = 0
    for path in sorted(pathlib.Path(folder).glob("ep_*.mjpeg")):
        if "_fail" in path.name:
            continue  # runs marked as failed are not driving worth copying
        rows = list(episodes.read(path))
        # Drop the standing-still frames before the first and after the last throttle press.
        moving = [i for i, r in enumerate(rows) if r[1]]
        if not moving:
            print(f"{path.name}: never moves, skipped")
            continue
        rows = rows[moving[0] : moving[-1] + 1]
        base = len(frames)
        times = [r[4] for r in rows]
        frames += [prepare(r[0]) for r in rows]
        steer += [steer_class(r[2], r[3]) for r in rows]
        throttle += [int(r[1] > 0) for r in rows]
        speeds += [r[1] for r in rows if r[1]]
        # A frame is an example only if the CHUNK frames from it are evenly spaced in time
        # (the camera can drop out) and all inside this episode.
        usable = [
            base + i
            for i in range(len(rows) - CHUNK + 1)
            if times[i + CHUNK - 1] - times[i] < (CHUNK - 1) * FRAME_MS * 1.6
        ]
        kept += 1
        (held if kept % 5 == 0 else train).extend(usable)
        print(f"{path.name}: {len(rows)} frames, {len(usable)} examples" + (", held out" if kept % 5 == 0 else ""))
    if not train:
        sys.exit(f"no usable episodes in {folder}/")
    speed = int(np.median(speeds))
    return np.stack(frames), np.array(steer), np.array(throttle), train, held, speed


def main(epochs=30, folder="data"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    frames, steer, throttle, train, held, speed = load(folder)
    frames = torch.from_numpy(frames).to(device)
    steer = torch.from_numpy(steer).to(device)
    throttle = torch.from_numpy(throttle).float().to(device)
    train, held = torch.tensor(train, device=device), torch.tensor(held, device=device)
    ahead = torch.arange(CHUNK, device=device)

    counts = torch.bincount(steer[train], minlength=3).float()
    print(
        f"{len(train)} training and {len(held)} held-out examples on {device}; steering "
        f"left/straight/right = {[int(c) for c in counts]}; throttle on in "
        f"{throttle[train].mean():.0%}; usual speed {speed} %"
    )
    # Turning frames are rare, so they count for more (capped so noise is not amplified).
    weights = (counts.sum() / (3 * counts.clamp(min=1))).clamp(max=5)

    def batch(index, augment):
        x = frames[index].float()
        s = steer[index[:, None] + ahead]
        t = throttle[index[:, None] + ahead]
        if augment:
            # Random brightness and contrast, so lighting changes matter less.
            gain = 1 + (torch.rand(len(index), 1, 1, 1, device=device) - 0.5) * 0.5
            shift = (torch.rand(len(index), 1, 1, 1, device=device) - 0.5) * 50
            x = (x * gain + shift).clamp(0, 255)
        return x, s, t

    def loss_of(net, index, augment):
        x, s, t = batch(index, augment)
        steer_logits, throttle_logits = net(x)
        loss = F.cross_entropy(steer_logits.reshape(-1, 3), s.reshape(-1), weight=weights)
        loss = loss + F.binary_cross_entropy_with_logits(throttle_logits, t)
        return loss, steer_logits.argmax(2), s, (throttle_logits > 0) == (t > 0.5)

    net = Net().to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * (len(train) // BATCH + 1)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=2e-3, total_steps=steps)
    best = float("inf")
    out = pathlib.Path(folder) / "model.pt"
    log = open(pathlib.Path(folder) / "train_log.csv", "w")
    log.write("epoch,train_loss,held_out_loss\n")
    for epoch in range(1, epochs + 1):
        net.train()
        total = 0.0
        order = train[torch.randperm(len(train), device=device)]
        for index in order.split(BATCH):
            loss = loss_of(net, index, augment=True)[0]
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item() * len(index)

        line = f"epoch {epoch:2d}  train loss {total / len(train):.3f}"
        score = total / len(train)
        if len(held):
            net.eval()
            with torch.no_grad():
                parts = [loss_of(net, index, augment=False) for index in held.split(1024)]
            score = sum(p[0].item() * len(p[1]) for p in parts) / len(held)
            guess = torch.cat([p[1] for p in parts])[:, 0]
            truth = torch.cat([p[2] for p in parts])[:, 0]
            throttle_ok = torch.cat([p[3] for p in parts])[:, 0].float().mean()
            recall = [
                f"{name} {(guess[truth == c] == c).float().mean():.0%}" if (truth == c).any() else f"{name} -"
                for name, c in (("left", LEFT), ("straight", STRAIGHT), ("right", RIGHT))
            ]
            line += (
                f"  held-out loss {score:.3f}  steering correct when driver went "
                f"{', '.join(recall)}  throttle {throttle_ok:.0%}"
            )
        log.write(f"{epoch},{total / len(train):.4f},{score:.4f}\n")
        log.flush()
        if score < best:
            best = score
            torch.save({"net": net.state_dict(), "speed": speed}, out)
            line += "  (saved)"
        print(line, flush=True)
    # The same network as an ONNX file, for ONNX Runtime on the Pi (pi/drive.cpp).
    net = Net()
    net.load_state_dict(torch.load(out, map_location="cpu")["net"])
    net.eval()
    onnx = out.with_suffix(".onnx")
    torch.onnx.export(
        net, torch.zeros(1, *frames.shape[1:], dtype=torch.uint8), str(onnx), dynamo=False,
        input_names=["frame"], output_names=["steer", "throttle"], opset_version=17,
    )
    print(f"best network is in {out} and {onnx}; drive at speed {speed}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30, *sys.argv[2:3])
