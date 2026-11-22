#!/usr/bin/bash
# Cross-compile pi/drive.cpp for the Pi. Run on the x86 Linux machine that has the QNX SDK
# and QNX's ONNX Runtime port installed into it (qnx-ports/build-files, ports/onnxruntime).
# Needs libturbojpeg.so.0 copied from the Pi (/usr/lib) next to drive.cpp.
set -e
source ~/qnx800/qnxsdp-env.sh
q++ -Vgcc_ntoaarch64le -std=c++17 -O2 -D_QNX_SOURCE drive.cpp -o drive \
    -I"$QNX_TARGET/usr/local/include" -I"$QNX_TARGET/usr/local/include/onnxruntime" \
    -L"$QNX_TARGET/aarch64le/usr/local/lib" -lonnxruntime -L. -l:libturbojpeg.so.0 -lsocket
ls -la drive
