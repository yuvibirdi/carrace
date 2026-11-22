#!/bin/sh
# Copy the latest network to the Pi and let the Pi drive the car by itself
# (ONNX Runtime on QNX, pi/drive.cpp). Extra arguments go to the program on the Pi:
#     host/drive_on_pi.sh [--dry] [--speed N] [--ahead N]
#     host/drive_on_pi.sh dreamer [--dry]      the Dreamer actor (data/dreamer.onnx) instead
# Close the teleop page first. Ctrl-C stops the car.
cd "$(dirname "$0")/.."
PI=${PI:-qnxpi.local}
export SSH_ASKPASS=$HOME/git/carrace-qnx-image/askpass.sh SSH_ASKPASS_REQUIRE=force
OPTS="-o ConnectTimeout=8 -o PreferredAuthentications=password -o PubkeyAuthentication=no"
MODEL=model.onnx
if [ "$1" = dreamer ]; then MODEL=dreamer.onnx; shift; fi
scp -q $OPTS data/$MODEL "qnxuser@$PI:/data/home/qnxuser/$MODEL" || exit 1
exec ssh -tt $OPTS "qnxuser@$PI" "cd /data/home/qnxuser && LD_LIBRARY_PATH=/data/home/qnxuser/ortlib ./drive --model $MODEL $*"
