"""Fetch and read the episodes recorded by the teleop app on the Pi.

Download everything that is new into data/:
    python3 host/episodes.py [pi-address]

The address defaults to qnxpi.local. If that name does not resolve on the phone
hotspot, pass the Pi's link-local address: fe80::ba27:ebff:fe74:f26c%en0

An episode file is the teleop stream saved as-is: for every frame a small
header block followed by the JPEG. read() yields the frames with their labels.
"""
import http.client
import pathlib
import sys

PORT = 8080
DATA = pathlib.Path("data")


def pull(pi="qnxpi.local"):
    """Download episodes that are missing or incomplete locally. Returns their paths."""
    DATA.mkdir(exist_ok=True)

    def get(path):
        conn = http.client.HTTPConnection(pi, PORT, timeout=10)
        conn.request("GET", path)
        return conn.getresponse()

    paths = []
    for line in get("/ep").read().decode().splitlines():
        name, size = line.split()
        dest = DATA / name
        # Fetched earlier, then marked or unmarked as a failed run on the Pi: follow the rename.
        other = DATA / (name.replace("_fail", "") if "_fail" in name else name[: -len(".mjpeg")] + "_fail.mjpeg")
        if other.exists():
            other.rename(dest)
        if not dest.exists() or dest.stat().st_size != int(size):
            print(f"downloading {name} ({int(size) / 1e6:.1f} MB)")
            with open(dest, "wb") as f:
                response = get("/ep/" + name)
                while chunk := response.read(1 << 20):
                    f.write(chunk)
        paths.append(dest)
    return paths


def read(path):
    """Yield (jpeg, throttle %, left, right, time ms) for each complete frame in an episode."""
    with open(path, "rb") as f:
        while f.readline().startswith(b"--frame"):
            headers = {}
            while line := f.readline().strip():
                name, _, value = line.partition(b": ")
                headers[name] = value
            size = int(headers.get(b"Content-Length", 0))
            jpeg = f.read(size)
            if not size or len(jpeg) < size:
                return  # the file was cut off mid-frame, e.g. by a power loss
            f.readline()
            throttle, left, right = (int(x) for x in headers[b"X-Action"].split())
            yield jpeg, throttle, left, right, int(headers[b"X-Time"])


if __name__ == "__main__":
    for path in pull(*sys.argv[1:2]):
        frames = list(read(path))
        seconds = (frames[-1][4] - frames[0][4]) / 1000 if len(frames) > 1 else 0
        moving = sum(1 for f in frames if f[1])
        print(f"{path.name}: {len(frames)} frames, {seconds:.0f} s, forward held in {moving}")
