"""Drive the car with the trained network, from this Mac over Wi-Fi.

    .venv/bin/python host/drive.py [--dry] [--speed N] [--ahead N] [--model PATH] [pi-address]

Close the teleop page first: an open page keeps sending "stop" and fights this script.
Ctrl-C stops the car. If this script or the link dies, the car stops by itself
after 0.3 s because the Pi stops receiving commands.

--dry    only print what the network would do; send nothing to the car
--speed  forward power in percent (default: the usual speed in the training data)
--ahead  which of the predicted actions to apply: 0 is "for this frame", higher
         values act earlier, which makes up for link delay (default 2, about 70 ms)
"""
import argparse
import http.client
import threading
import time

import torch

from model import LEFT, RIGHT, Net, prepare

PORT = 8080
latest, latest_seq = None, 0
arrived = threading.Condition()


def watch(pi):
    """Keep only the newest camera frame, so the network never works on a stale one."""
    global latest, latest_seq
    stream = http.client.HTTPConnection(pi, PORT, timeout=5)
    stream.request("GET", "/stream")
    f = stream.getresponse()
    while f.readline().startswith(b"--frame"):
        size = 0
        while line := f.readline().strip():
            if line.startswith(b"Content-Length:"):
                size = int(line.split(b":")[1])
        jpeg = f.read(size)
        f.readline()
        with arrived:
            latest, latest_seq = jpeg, latest_seq + 1
            arrived.notify()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pi", nargs="?", default="qnxpi.local")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--speed", type=int)
    p.add_argument("--ahead", type=int, default=2)
    p.add_argument("--model", default="data/model.pt")
    args = p.parse_args()

    saved = torch.load(args.model, map_location="cpu")
    net = Net()
    net.load_state_dict(saved["net"])
    net.eval()
    speed = args.speed or saved["speed"]
    print(f"speed {speed} %, using predicted action {args.ahead}" + (" (dry run)" if args.dry else ""))

    threading.Thread(target=watch, args=(args.pi,), daemon=True).start()
    car = http.client.HTTPConnection(args.pi, PORT, timeout=2)

    def send(bits):
        car.request("GET", f"/s/{bits}/{speed}")
        car.getresponse().read()

    seq, count, shown = 0, 0, time.monotonic()
    try:
        while True:
            with arrived:
                if not arrived.wait_for(lambda: latest_seq != seq, timeout=1):
                    raise SystemExit("no camera frames for 1 s, stopping")
                jpeg, seq = latest, latest_seq
            with torch.no_grad():
                steer, throttle = net(torch.from_numpy(prepare(jpeg))[None])
            turn = steer[0, args.ahead].argmax().item()
            bits = f"{int(throttle[0, args.ahead] > 0)}{int(turn == LEFT)}{int(turn == RIGHT)}"
            if not args.dry:
                send(bits)
            count += 1
            if time.monotonic() - shown >= 1:
                print(f"{count} decisions/s, now: forward {bits[0]} left {bits[1]} right {bits[2]}", flush=True)
                count, shown = 0, time.monotonic()
    finally:
        if not args.dry:
            car.close()
            send("000")


if __name__ == "__main__":
    main()
