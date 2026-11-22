"""Teleop app for the car's QNX Pi: live camera view plus keyboard/touch driving.

Run on the Pi:  python3 teleop.py
Then open http://qnxpi.local:8080

Recording: the Record button (or R) starts an episode and Stop ends it; drive
one full run of the track per episode. Each episode is one file on the Pi,
~/episodes/ep_NNNN.mjpeg, holding every published frame in the same format as
/stream: a JPEG plus X-Action ("throttle% left right" applied when it was
captured) and X-Time (ms). /ep lists the files and /ep/<name> downloads one.

Pi BCM GPIO -> what it does on the car (found by testing), each through ~1k
into an MX1919 input, plus Pi GND -> board GND:
  GPIO19 forward    GPIO23 left    GPIO24 right
There is no reverse: that input pin on the MX1919 broke off.
Forward is PWM, so speed is adjustable; left and right are plain on/off.

The QNX image has no Python camera module, so the camera is read through QNX's
C camera library (libcamapi) with ctypes. The stereo camera sends JPEG frames
holding two views side by side; libturbojpeg cuts out the left one losslessly.

The camera needs the MJPEG entry for SENSOR_UNIT_5 in
/etc/config/sensor/sensor_rpi5.conf (default_video_format = jpeg, 640x240 @ 60,
create_window = false, memory_allocator_type = from_sensor); the stock entry is
kept next to it as sensor_rpi5.conf.orig.
"""
import ctypes
import json
import os
import re
import shutil
import signal
import socket
import sys
import threading
import time
from ctypes import byref, c_int, c_int32, c_int64, c_uint32, c_uint64, c_ulong, c_void_p
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rpi_gpio as GPIO

PINS = (19, 23, 24)  # forward, left, right
OFF = "0" * len(PINS)
TIMEOUT_S = 0.3  # all pins go low if the browser stops sending
PWM_HZ = 500  # forward is pulsed at this rate; its duty cycle sets the speed
KICK_S = 0.15  # full-power pulse when starting from rest, so low speeds still get the car moving
PORT = 8080  # 8000 is taken on the QNX image
EPISODES = os.path.expanduser("~/episodes")

CAMERA_UNIT = 5  # the USB camera entry in /etc/config/sensor/sensor_rpi5.conf
EYE_W, EYE_H = 320, 240  # the camera sends two of these side by side; the left one is published
KEEP_EVERY = 2  # the camera runs at 60 fps; publishing every 2nd frame gives 30
FLIP_180 = False  # set True if the camera is mounted upside down

# Values from the QNX SDP 8.0 headers camera/camera_defs.h and camera_config.h.
CAMERA_MODE = 1 | 2 | 4  # PREAD | PWRITE | DREAD
CAMERA_EVENTMODE_READONLY = 0
CAMERA_FRAMETYPE_JPEG = 48  # not in the SDP headers; what the Pi image reports for jpeg
# From turbojpeg.h.
TJSAMP_444, TJFLAG_NOREALLOC = 0, 1024
TJXOP_NONE, TJXOP_ROT180, TJXOPT_CROP = 0, 6, 4


class Region(ctypes.Structure):  # tjregion
    _fields_ = [("x", c_int), ("y", c_int), ("w", c_int), ("h", c_int)]


class Transform(ctypes.Structure):  # tjtransform
    _fields_ = [("r", Region), ("op", c_int), ("options", c_int), ("data", c_void_p), ("filter", c_void_p)]


class FrameDesc(ctypes.Union):  # camera_framedesc_t: 256 bytes; for jpeg it starts with the byte count
    _fields_ = [("bufsize", c_uint64), ("_size", c_uint64 * 32)]


class CameraBuffer(ctypes.Structure):  # camera_buffer_t
    _fields_ = [
        ("frametype", c_int),
        ("framesize", c_uint64),
        ("framebuf", c_void_p),
        ("framemetasize", c_uint64),
        ("framemeta", c_void_p),
        ("frametimestamp", c_int64),
        ("frameorientation", c_int32),
        ("reserved", c_uint32 * 4),
        ("framedesc", FrameDesc),
    ]


# The library fills in the whole struct, so a shorter one corrupts memory.
assert ctypes.sizeof(CameraBuffer) == 328

PAGE = b"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>carrace teleop</title>
<style>
  body { margin: 0; background: #111; color: #ddd; font: 15px system-ui, sans-serif;
         display: flex; flex-direction: column; align-items: center; }
  header { width: 100%; max-width: 640px; box-sizing: border-box; padding: 10px 12px;
           display: flex; justify-content: space-between; align-items: baseline; }
  h1 { font-size: 16px; margin: 0; }
  #status.lost { color: #f55; font-weight: bold; }
  #video { width: 100%; max-width: 640px; aspect-ratio: 4 / 3; background: #000; display: block; }
  #pad { display: flex; gap: 12px; margin: 16px; }
  #pad button { width: 96px; height: 72px; font-size: 24px; line-height: 1.1; border-radius: 12px;
           border: 2px solid #444; background: #222; color: #ddd;
           touch-action: none; user-select: none; -webkit-user-select: none; }
  #pad button small { font-size: 13px; color: #888; }
  #pad button.on { background: #2a6; border-color: #3c8; color: #fff; }
  #pad button.on small { color: #dfd; }
  #rec { display: flex; gap: 10px; align-items: center; margin: 0 0 14px; }
  #rec button { height: 40px; padding: 0 16px; font-size: 15px; border-radius: 10px;
                border: 2px solid #444; background: #222; color: #ddd; }
  #rec button.on { background: #c33; border-color: #f55; color: #fff; }
  #tally { margin: 0 0 6px; color: #ccc; }
  #eps { border-collapse: collapse; margin: 0 0 14px; font-size: 14px; }
  #eps td { padding: 3px 10px; border-top: 1px solid #333; }
  #eps tr.failed td { color: #f77; }
  #eps button { border-radius: 8px; border: 1px solid #555; background: #222; color: #ddd; padding: 2px 10px; }
  #speed { display: flex; gap: 10px; align-items: center; margin: 0 0 14px; }
  #speed input { width: 220px; }
  p { color: #888; margin: 0 12px 16px; text-align: center; }
</style>
<header><h1>carrace teleop</h1><span id="status">connecting</span></header>
<img id="video" src="/stream" alt="camera view">
<div id="pad">
  <button data-i="1">&#9664;<br><small>A</small></button>
  <button data-i="0">&#9650;<br><small>W</small></button>
  <button data-i="2">&#9654;<br><small>D</small></button>
</div>
<div id="speed">speed <input id="speedIn" type="range" min="30" max="100" step="1" value="30">
  <b id="speedOut">30%</b></div>
<div id="rec"><button id="recBtn">Record (R)</button><span id="recInfo"></span>
  <button id="failBtn">Failed run (F)</button>
  <button id="discardBtn">Discard last</button></div>
<p>Hold W / A / D, the arrow keys, or the buttons. Keys 1-5 or the slider set forward speed.
   The car stops when you let go. Record one full run of the track per episode.
   F stops the car and marks the current (or last) episode as a failed run.</p>
<div id="tally"></div>
<table id="eps"></table>
<script>
const keys = {w: 0, a: 1, d: 2, arrowup: 0, arrowleft: 1, arrowright: 2};
const held = [0, 0, 0];
const buttons = [...document.querySelectorAll("#pad button")];
const statusEl = document.getElementById("status");
const video = document.getElementById("video");
const speedIn = document.getElementById("speedIn");
const speedOut = document.getElementById("speedOut");
const recBtn = document.getElementById("recBtn");
const recInfo = document.getElementById("recInfo");
let lastOk = performance.now();
let recording = false;

function showRec(s) {
  recording = s.rec !== "";
  recBtn.textContent = recording ? "Stop (R)" : "Record (R)";
  recBtn.classList.toggle("on", recording);
  recInfo.textContent = (recording ? "REC " + s.rec + ", " + s.frames + " frames, " : "") +
    s.episodes + " saved" + (s.failed ? " (" + s.failed + " failed)" : "") + ", " + s.free_gb + " GB free";
}
// The episode list: speed is read from each recording; the button marks or unmarks a failed run.
function showEpisodes(list) {
  const bySpeed = {};
  list.forEach(e => {
    const t = bySpeed[e.speed] = bySpeed[e.speed] || [0, 0];
    t[e.failed ? 1 : 0]++;
  });
  document.getElementById("tally").textContent = Object.keys(bySpeed).sort((a, b) => a - b)
    .map(v => v + " %: " + bySpeed[v][0] + " ok, " + bySpeed[v][1] + " failed").join("   |   ");
  const table = document.getElementById("eps");
  table.replaceChildren(...list.slice().reverse().map(e => {
    const row = table.insertRow();
    row.className = e.failed ? "failed" : "";
    row.insertCell().textContent = e.name.slice(0, 7);
    row.insertCell().textContent = e.speed + " %";
    row.insertCell().textContent = e.seconds + " s";
    row.insertCell().textContent = e.failed ? "FAILED" : "ok";
    const button = document.createElement("button");
    button.textContent = e.failed ? "undo" : "mark failed";
    button.onclick = () => fetch("/ep/" + e.name + "/fail").then(loadEpisodes);
    row.insertCell().append(button);
    return row;
  }));
}
function loadEpisodes() { fetch("/eps").then(r => r.json()).then(showEpisodes).catch(() => {}); }
loadEpisodes();
setInterval(loadEpisodes, 3000);
function rec(cmd) {
  fetch("/rec/" + cmd).then(r => r.json()).then(s => { showRec(s); loadEpisodes(); }).catch(() => {});
}
recBtn.onclick = () => rec(recording ? "stop" : "start");
document.getElementById("failBtn").onclick = () => rec("fail");
document.getElementById("discardBtn").onclick = () => {
  if (confirm("Delete the most recent episode?")) rec("discard");
};

// Drive commands are only sent while a key is held (and once on release), so the page
// can stay open to watch, record and mark failures while the network drives the car.
function send(path) {
  const t0 = performance.now();
  fetch(path || "/s/" + held.join("") + "/" + speedIn.value).then(r => r.json()).then(s => {
    lastOk = performance.now();
    statusEl.textContent = "link " + Math.round(lastOk - t0) + " ms, camera " + s.fps + " fps";
    statusEl.className = "";
    showRec(s);
  }).catch(() => {});
}
function set(i, v) {
  if (held[i] === v) return;
  held[i] = v;
  buttons.forEach(b => b.classList.toggle("on", held[b.dataset.i] === 1));
  send();
}
function setSpeed(v) { speedIn.value = v; speedOut.textContent = speedIn.value + "%"; send(); }
speedIn.oninput = () => setSpeed(speedIn.value);
onkeydown = e => {
  const k = e.key.toLowerCase();
  if (k in keys) { e.preventDefault(); set(keys[k], 1); }
  else if (k >= "1" && k <= "5") setSpeed(Math.round(30 + (k - 1) * 17.5));  // 30..100
  else if (k === "r" && !e.repeat) rec(recording ? "stop" : "start");
  else if (k === "f" && !e.repeat) rec("fail");
};
onkeyup = e => { const k = e.key.toLowerCase(); if (k in keys) set(keys[k], 0); };
onblur = () => held.forEach((_, i) => set(i, 0));
buttons.forEach(b => {
  const i = Number(b.dataset.i);
  b.onpointerdown = e => { b.setPointerCapture(e.pointerId); set(i, 1); };
  b.onpointerup = b.onpointercancel = () => set(i, 0);
});
video.onerror = () => setTimeout(() => { video.src = "/stream?" + Date.now(); }, 1000);
setInterval(() => {
  send(held.includes(1) ? "" : "/status");
  if (performance.now() - lastOk > 600) {
    statusEl.textContent = "LINK LOST";
    statusEl.className = "lost";
  }
}, 100);
</script>
</html>
"""

last_cmd = 0.0
halt_until = 0.0  # after a run is marked failed, drive commands are refused until then
frame = b""  # latest camera frame as a ready-to-send multipart part (headers + JPEG)
frame_seq = 0
frame_time = 0.0
frame_cond = threading.Condition()
cam_fps = 0.0
applied = "0 0 0"  # "throttle% left right" right now
target = 0  # forward throttle percent currently asked for
kick_end = 0.0  # until this time forward is held at full power
drive_lock = threading.Lock()
rec = None  # open episode file while recording
rec_name = ""
rec_frames = 0
rec_lock = threading.Lock()


def write(bits, speed=100):
    """Apply forward/left/right bits, with forward running at `speed` percent."""
    global applied, target, kick_end
    throttle = speed if bits[0] == "1" else 0
    state = f"{throttle} {bits[1]} {bits[2]}"
    with drive_lock:
        if state == applied:
            return
        if throttle != target:
            if not throttle:
                kick_end = 0.0
                forward.ChangeDutyCycle(0)
            elif not target:  # starting from rest: kick, then settle() drops to the asked speed
                kick_end = time.monotonic() + KICK_S
                forward.ChangeDutyCycle(100)
                threading.Timer(KICK_S, settle).start()
            elif time.monotonic() >= kick_end:
                forward.ChangeDutyCycle(throttle)
            target = throttle
        GPIO.output(PINS[1], GPIO.HIGH if bits[1] == "1" else GPIO.LOW)
        GPIO.output(PINS[2], GPIO.HIGH if bits[2] == "1" else GPIO.LOW)
        applied = state


def settle():
    """End of the start-up kick: drop to the speed that was asked for."""
    with drive_lock:
        if target and time.monotonic() >= kick_end:
            forward.ChangeDutyCycle(target)


def watchdog():
    while True:
        time.sleep(0.05)
        if time.monotonic() - last_cmd > TIMEOUT_S:
            write(OFF)


def episodes():
    return sorted(f for f in os.listdir(EPISODES) if f.startswith("ep_") and f.endswith(".mjpeg"))


details_cache = {}


def details(name):
    """Speed (the most used throttle percent) and driving time of a finished episode,
    read from the labels on its frames."""
    path = os.path.join(EPISODES, name)
    key, size = name.replace("_fail", ""), os.path.getsize(path)
    if details_cache.get(key, (None,))[0] != size:
        with open(path, "rb") as f:
            labels = re.findall(rb"X-Action: (\d+) \d \d\r\nX-Time: (\d+)", f.read())
        moving = [(int(throttle), int(ms)) for throttle, ms in labels if throttle != b"0"]
        speeds = [throttle for throttle, _ in moving]
        speed = max(set(speeds), key=speeds.count) if speeds else 0
        seconds = round((moving[-1][1] - moving[0][1]) / 1000, 1) if moving else 0
        details_cache[key] = (size, speed, seconds)
    return {"name": name, "speed": details_cache[key][1], "seconds": details_cache[key][2], "failed": "_fail" in name}


def toggle_failed(name):
    with rec_lock:
        if rec and name == rec_name + ".mjpeg":
            return  # still recording: F marks that one
        old = os.path.join(EPISODES, name)
        new = name.replace("_fail", "") if "_fail" in name else name[: -len(".mjpeg")] + "_fail.mjpeg"
        os.rename(old, os.path.join(EPISODES, new))


def record(cmd):
    """start: begin a new episode; stop: end it; discard: delete the most recent one;
    fail: end it, mark the most recent one as a failed run and stop the car."""
    global rec, rec_name, rec_frames, halt_until
    with rec_lock:
        if rec:
            rec.close()
            rec = None
        done = episodes()
        if cmd == "start":
            rec_name = f"ep_{int(done[-1][3:7]) + 1 if done else 1:04d}"
            rec_frames = 0
            rec = open(os.path.join(EPISODES, rec_name + ".mjpeg"), "wb")
        elif cmd == "discard" and done:
            os.remove(os.path.join(EPISODES, done[-1]))
        elif cmd == "fail":
            halt_until = time.monotonic() + 1.5
            if done and "_fail" not in done[-1]:
                old = os.path.join(EPISODES, done[-1])
                os.rename(old, old[: -len(".mjpeg")] + "_fail.mjpeg")
    if cmd == "fail":
        write(OFF)


def status():
    fps = cam_fps if time.monotonic() - frame_time < 1.5 else 0  # 0 once the camera stalls
    return json.dumps({
        "fps": round(fps),
        "rec": rec_name if rec else "",
        "frames": rec_frames,
        "episodes": len(episodes()),
        "failed": sum("_fail" in f for f in episodes()),
        "halt": int(time.monotonic() < halt_until),  # tells pi/drive.cpp to quit
        "free_gb": round(shutil.disk_usage(EPISODES).free / 1e9, 1),
    }).encode()


def camera():
    global frame, frame_seq, frame_time, cam_fps, rec_frames
    cam = ctypes.CDLL("libcamapi.so.1")
    tj = ctypes.CDLL("libturbojpeg.so.0")
    tj.tjInitTransform.restype = c_void_p
    tj.tjBufSize.restype = c_ulong
    cropper = c_void_p(tj.tjInitTransform())
    jpeg = ctypes.create_string_buffer(tj.tjBufSize(2 * EYE_W, EYE_H, TJSAMP_444))
    # Lossless cut of the left view. After a 180-degree turn it sits on the right.
    cut = Transform(
        Region(EYE_W if FLIP_180 else 0, 0, EYE_W, EYE_H),
        TJXOP_ROT180 if FLIP_180 else TJXOP_NONE,
        TJXOPT_CROP,
        None,
        None,
    )

    def check(what, err):
        if err:
            raise RuntimeError(f"camera: {what} failed with error {err}")

    handle, key = c_int32(-1), c_int32(-1)
    no_event = (ctypes.c_char * 32)()  # zeroed struct sigevent = SIGEV_NONE: we poll instead
    # The camera may be unplugged or still registering after boot, so keep trying.
    while (err := cam.camera_open(CAMERA_UNIT, CAMERA_MODE, byref(handle))) != 0:
        print(f"camera: open failed with error {err}, retrying", flush=True)
        time.sleep(2)
    # Frames are fetched by polling from this thread. Letting the library call
    # back into Python from its own thread crashes the interpreter.
    check(
        "enable frame access",
        cam.camera_enable_viewfinder_event(handle, CAMERA_EVENTMODE_READONLY, byref(key), no_event),
    )
    check("start", cam.camera_start_viewfinder(handle, None, None, None))

    buf = CameraBuffer()
    seen, count, t0 = 0, 0, time.monotonic()
    while True:
        if cam.camera_get_viewfinder_buffers(handle, key, byref(buf), None) != 0:
            time.sleep(0.002)  # no new frame yet
            continue
        seen += 1
        wanted = buf.frametype == CAMERA_FRAMETYPE_JPEG and seen % KEEP_EVERY == 0
        stereo = ctypes.string_at(buf.framebuf, buf.framedesc.bufsize) if wanted else None
        cam.camera_return_buffer(handle, byref(buf))
        if stereo is None:
            continue

        dst, size = c_void_p(ctypes.addressof(jpeg)), c_ulong(len(jpeg))
        if tj.tjTransform(
            cropper, stereo, c_ulong(len(stereo)), 1,
            byref(dst), byref(size), byref(cut), TJFLAG_NOREALLOC,
        ) != 0:
            continue

        now = time.monotonic()
        # X-Action and X-Time (ms on the Pi's clock) label the frame for training.
        part = (
            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n"
            b"X-Action: %b\r\nX-Time: %d\r\n\r\n%b\r\n"
            % (size.value, applied.encode(), now * 1000, ctypes.string_at(jpeg, size.value))
        )
        with frame_cond:
            frame, frame_seq, frame_time = part, frame_seq + 1, now
            frame_cond.notify_all()
        with rec_lock:
            if rec:
                rec.write(part)
                rec_frames += 1
        count += 1
        if now - t0 >= 1.0:
            cam_fps, count, t0 = count / (now - t0), 0, now


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global last_cmd
        bits, _, speed = self.path[3:].partition("/")
        speed = speed or "100"
        if self.path == "/":
            self.reply(200, PAGE)
        elif self.path.startswith("/stream"):
            self.stream()
        elif self.path == "/status":
            self.reply(200, status())
        elif self.path in ("/rec/start", "/rec/stop", "/rec/discard", "/rec/fail"):
            record(self.path[5:])
            self.reply(200, status())
        elif self.path == "/eps":
            with rec_lock:
                recording = rec_name + ".mjpeg" if rec else ""
            self.reply(200, json.dumps([details(f) for f in episodes() if f != recording]).encode())
        elif self.path.endswith("/fail") and self.path[4:-5] in episodes():
            toggle_failed(self.path[4:-5])
            self.reply(200, status())
        elif self.path == "/ep":
            sizes = (f"{f} {os.path.getsize(os.path.join(EPISODES, f))}" for f in episodes())
            self.reply(200, "\n".join(sizes).encode())
        elif self.path.startswith("/ep/") and self.path[4:] in episodes():
            self.send_file(os.path.join(EPISODES, self.path[4:]))
        elif (
            self.path.startswith("/s/")
            and len(bits) == len(PINS)
            and set(bits) <= {"0", "1"}
            and speed.isdigit()
            and int(speed) <= 100
        ):
            if time.monotonic() >= halt_until:
                last_cmd = time.monotonic()
                write(bits, int(speed))
            self.reply(200, status())
        else:
            self.reply(404, b"not found")

    def reply(self, status, body):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        self.send_response(200)
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seq = 0
        try:
            while True:
                with frame_cond:
                    frame_cond.wait_for(lambda: frame_seq != seq)
                    seq, part = frame_seq, frame
                self.wfile.write(part)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser closed the stream

    def log_message(self, format, *args):  # 10 requests/s would flood the terminal
        pass


class Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6  # a phone hotspot may give the laptop IPv6 only

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)  # IPv4 too
        super().server_bind()


os.makedirs(EPISODES, exist_ok=True)
GPIO.setmode(GPIO.BCM)
# Default PWM mode on purpose: passing MODE_MS explicitly gives no output on this Pi 5.
forward = GPIO.PWM(PINS[0], PWM_HZ)
forward.start(0)
for pin in PINS[1:]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # so `slay` still runs the cleanup below
threading.Thread(target=watchdog, daemon=True).start()
threading.Thread(target=camera, daemon=True).start()
try:
    Server(("::", PORT), Handler).serve_forever()
finally:
    write(OFF)  # QNX's GPIO.cleanup() does not reset pins, so do it here
    record("stop")
