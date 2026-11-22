// Drive the car with the trained network on the Pi itself, under QNX.
//
// The network runs in ONNX Runtime (QNX's open-source port, github.com/qnx-ports).
// This program is the on-car twin of host/drive.py: it reads camera frames from the
// teleop app's stream on this machine, runs the network on each one and sends the
// resulting command back to the teleop app, which sets the motor pins and records.
//
//     LD_LIBRARY_PATH=~/lib ./drive [--dry] [--speed N] [--ahead N] [--model model.onnx]
//
// With the Dreamer actor (--model dreamer.onnx, from host/dreamer.py) the network also
// picks the throttle level and keeps a memory between frames; --speed and --ahead are unused.
//     LD_LIBRARY_PATH=~/lib ./drive --check frame.jpg    # print the raw outputs for one picture
//
// Close the teleop page first: an open page keeps sending "stop" and fights this.
// Ctrl-C stops the car. If this program dies, the car stops by itself after 0.3 s.
//
// Build (on the x86 machine with the QNX SDK): see pi/build_drive.sh
#include <onnxruntime_cxx_api.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

// libturbojpeg is on the QNX image but its header is not in the SDK.
extern "C" {
void *tjInitDecompress(void);
int tjDecompress2(void *handle, const unsigned char *jpeg, unsigned long size, unsigned char *dst,
                  int width, int pitch, int height, int pixelFormat, int flags);
char *tjGetErrorStr2(void *handle);
}
static const int TJPF_RGB = 0;

static const int W = 160, H = 120, CHUNK = 10;  // must match host/model.py
static const int DETER = 256, STOCH = 32, LEVELS = 6;  // must match host/dreamer.py
static const int SPEEDS[LEVELS] = {0, 60, 70, 80, 90, 100};
static const int PORT = 8080;
static volatile sig_atomic_t stopping = 0;

static int connect_local() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (fd < 0 || connect(fd, (sockaddr *)&addr, sizeof addr) != 0) {
        perror("cannot reach the teleop app on port 8080");
        exit(1);
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    return fd;
}

// Sends a drive command. Returns false once the teleop app says a run was marked failed (F key).
static bool command(int forward, int left, int right, int speed) {
    char request[64];
    int n = snprintf(request, sizeof request, "GET /s/%d%d%d/%d HTTP/1.0\r\n\r\n", forward, left, right, speed);
    int fd = connect_local();
    std::string reply;
    char part[512];
    if (write(fd, request, n) == n)
        for (ssize_t got; (got = read(fd, part, sizeof part)) > 0;) reply.append(part, got);
    close(fd);
    return reply.find("\"halt\": 1") == std::string::npos;
}

// Buffered reading from the stream socket.
struct Reader {
    int fd;
    std::vector<char> buf = std::vector<char>(1 << 16);
    size_t pos = 0, end = 0;

    bool fill() {
        ssize_t n = read(fd, buf.data(), buf.size());
        pos = 0;
        end = n > 0 ? n : 0;
        return n > 0;
    }
    bool line(std::string &out) {
        out.clear();
        for (;;) {
            if (pos == end && !fill()) return false;
            char c = buf[pos++];
            if (c == '\n') break;
            if (c != '\r') out += c;
        }
        return true;
    }
    bool bytes(std::string &out, size_t n) {
        out.clear();
        while (out.size() < n) {
            if (pos == end && !fill()) return false;
            size_t take = std::min(n - out.size(), end - pos);
            out.append(buf.data() + pos, take);
            pos += take;
        }
        return true;
    }
    bool more_waiting() {
        pollfd p{fd, POLLIN, 0};
        return pos < end || poll(&p, 1, 0) > 0;
    }
};

static double now() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    bool dry = false;
    int speed = 60, ahead = 2;
    std::string model = "model.onnx", check;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--dry") dry = true;
        else if (a == "--speed" && i + 1 < argc) speed = atoi(argv[++i]);
        else if (a == "--ahead" && i + 1 < argc) ahead = atoi(argv[++i]);
        else if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--check" && i + 1 < argc) check = argv[++i];
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (ahead < 0 || ahead >= CHUNK || speed < 0 || speed > 100) { fprintf(stderr, "bad --ahead or --speed\n"); return 2; }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "drive");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(2);
    Ort::Session session(env, model.c_str(), options);
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<unsigned char> rgb(W * H * 3);
    const int64_t shape[4] = {1, H, W, 3};
    std::vector<Ort::Value> feed;
    feed.push_back(Ort::Value::CreateTensor<uint8_t>(memory, rgb.data(), rgb.size(), shape, 4));
    // The Dreamer actor also takes its state from the frame before, the action it chose
    // then, and a flag for the first frame of the run.
    const bool dreamer = session.GetInputCount() == 5;
    std::vector<float> h(DETER), z(STOCH), action(3 + LEVELS), first(1, 1);
    const int64_t h_shape[2] = {1, DETER}, z_shape[2] = {1, STOCH}, a_shape[2] = {1, 3 + LEVELS}, f_shape[2] = {1, 1};
    if (dreamer) {
        feed.push_back(Ort::Value::CreateTensor<float>(memory, h.data(), h.size(), h_shape, 2));
        feed.push_back(Ort::Value::CreateTensor<float>(memory, z.data(), z.size(), z_shape, 2));
        feed.push_back(Ort::Value::CreateTensor<float>(memory, action.data(), action.size(), a_shape, 2));
        feed.push_back(Ort::Value::CreateTensor<float>(memory, first.data(), first.size(), f_shape, 2));
    }
    const char *inputs[] = {"frame", "h", "z", "action", "first"};
    const char *plain_outputs[] = {"steer", "throttle"}, *dreamer_outputs[] = {"steer", "level", "h_next", "z_next"};
    void *decoder = tjInitDecompress();

    // Decode at half size inside the JPEG decoder, exactly like host/model.py prepare().
    // For the Dreamer actor, "throttle" is one score per throttle level.
    auto decide = [&](const std::string &jpeg, const float *&steer, const float *&throttle,
                      std::vector<Ort::Value> &result) {
        if (tjDecompress2(decoder, (const unsigned char *)jpeg.data(), jpeg.size(), rgb.data(), W, 0, H, TJPF_RGB, 0) != 0) {
            fprintf(stderr, "bad picture: %s\n", tjGetErrorStr2(decoder));
            return false;
        }
        result = session.Run(Ort::RunOptions{nullptr}, inputs, feed.data(), feed.size(),
                             dreamer ? dreamer_outputs : plain_outputs, dreamer ? 4 : 2);
        steer = result[0].GetTensorData<float>();
        throttle = result[1].GetTensorData<float>();
        if (dreamer) {  // keep the new state for the next frame
            std::copy_n(result[2].GetTensorData<float>(), DETER, h.begin());
            std::copy_n(result[3].GetTensorData<float>(), STOCH, z.begin());
            first[0] = 0;
        }
        return true;
    };

    std::vector<Ort::Value> result;
    const float *steer, *throttle;
    if (!check.empty()) {
        FILE *f = fopen(check.c_str(), "rb");
        if (!f) { perror(check.c_str()); return 1; }
        std::string jpeg;
        char chunk[65536];
        for (size_t n; (n = fread(chunk, 1, sizeof chunk, f)) > 0;) jpeg.append(chunk, n);
        fclose(f);
        if (!decide(jpeg, steer, throttle, result)) return 1;
        printf("pixels %d %d %d %d\n", rgb[0], rgb[1], rgb[2], rgb[rgb.size() / 2]);
        if (dreamer) {
            printf("steer %.4f %.4f %.4f\nlevel", steer[0], steer[1], steer[2]);
            for (int i = 0; i < LEVELS; i++) printf(" %.4f", throttle[i]);
            printf("\nstate %.4f %.4f %.4f %.4f\n", h[0], h[DETER - 1], z[0], z[STOCH - 1]);
            return 0;
        }
        for (int i = 0; i < CHUNK; i++)
            printf("%d: steer %.4f %.4f %.4f throttle %.4f\n", i, steer[i * 3], steer[i * 3 + 1], steer[i * 3 + 2], throttle[i]);
        return 0;
    }

    signal(SIGINT, [](int) { stopping = 1; });
    signal(SIGTERM, [](int) { stopping = 1; });
    if (dreamer) printf("Dreamer actor, it picks the throttle level itself%s\n", dry ? " (dry run)" : "");
    else printf("speed %d %%, using predicted action %d%s\n", speed, ahead, dry ? " (dry run)" : "");
    fflush(stdout);

    Reader stream{connect_local()};
    timeval patience{1, 0};  // a camera that goes quiet for 1 s ends the run
    setsockopt(stream.fd, SOL_SOCKET, SO_RCVTIMEO, &patience, sizeof patience);
    const char *get = "GET /stream HTTP/1.0\r\n\r\n";
    if (write(stream.fd, get, strlen(get)) < 0) return 1;
    std::string text, jpeg;
    while (stream.line(text) && !text.empty()) {  // skip the reply headers
    }

    int count = 0, forward = 0, left = 0, right = 0;
    double shown = now(), spent = 0;
    while (!stopping && stream.line(text)) {
        if (text.rfind("--frame", 0) != 0) continue;
        size_t size = 0;
        while (stream.line(text) && !text.empty())
            if (text.rfind("Content-Length:", 0) == 0) size = atoi(text.c_str() + 15);
        if (!size || !stream.bytes(jpeg, size) || !stream.line(text)) break;  // picture, then its end-of-line
        if (stream.more_waiting()) continue;  // a newer frame is already here: never act on a stale one

        double t0 = now();
        if (!decide(jpeg, steer, throttle, result)) continue;
        int turn, percent;
        if (dreamer) {
            turn = std::max_element(steer, steer + 3) - steer;
            int level = std::max_element(throttle, throttle + LEVELS) - throttle;
            percent = SPEEDS[level];
            std::fill(action.begin(), action.end(), 0.f);  // what the next frame is told was done
            action[turn] = action[3 + level] = 1;
        } else {
            const float *s = steer + ahead * 3;
            turn = s[0] > s[1] ? (s[0] > s[2] ? 0 : 2) : (s[1] >= s[2] ? 1 : 2);
            percent = throttle[ahead] > 0 ? speed : 0;
        }
        forward = percent > 0, left = turn == 0, right = turn == 2;
        spent += now() - t0;
        if (!dry && !command(forward, left, right, forward ? percent : speed)) {
            printf("run marked as failed on the page, stopping\n");
            break;
        }
        count++;
        if (now() - shown >= 1) {
            printf("%d decisions/s, %.1f ms each, now: forward %d left %d right %d at %d %%\n",
                   count, 1000 * spent / count, forward, left, right, percent);
            fflush(stdout);
            count = 0, spent = 0, shown = now();
        }
    }
    if (!dry) command(0, 0, 0, speed);
    return 0;
}
