#!/usr/bin/env bash
set -e
echo "START llvm $(date)"
sudo apt-get update -qq
sudo apt-get install -y -qq lsb-release wget software-properties-common gnupg >/dev/null
cd /tmp
wget -q https://apt.llvm.org/llvm.sh -O llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 21 >/dev/null 2>&1 || true
# explicit: llc (llvm-21), libclang for bindgen, clang for resource-dir stddef.h
sudo apt-get install -y -qq llvm-21 clang-21 libclang-common-21-dev >/dev/null
echo "--- versions ---"
/usr/bin/llc-21 --version 2>/dev/null | grep -iE "LLVM version|nvptx" || echo "NO llc-21"
/usr/bin/clang-21 --version 2>/dev/null | head -1 || echo "NO clang-21"
echo "--- C: headroom after ---"; df -h / | tail -1
echo "EXIT=$? DONE llvm $(date)"
