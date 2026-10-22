#!/bin/bash
# flash_board.sh -- put the lab firmware and a model on a real
# STM32F4DISCOVERY, then (optionally) stream images to it.
#
#   tools/flash_board.sh                              # firmware + submission/submission.pte
#   tools/flash_board.sh runs/x/submission.pte        # firmware + another model
#   tools/flash_board.sh runs/x/submission.pte --model-only
#   python3 tools/send_images.py --port /dev/ttyUSB0 --val-n 500    # then run it
#
# The firmware (sectors 0-7, 0x08000000) only needs flashing once; the model
# (header + .pte, sectors 8-11, 0x08080000) is written separately, so
# --model-only is enough when you change models.
#
# Wiring: a 3.3 V USB-serial adapter on PA2 (board TX -> adapter RX), PA3
# (board RX <- adapter TX) and GND; 115200 8N1.
set -euo pipefail
cd "$(dirname "$0")/.."      # paths are relative to the lab repo root
PTE="${1:-submission/submission.pte}"
BIN="$(mktemp --suffix=.bin)"
trap 'rm -f "$BIN"' EXIT
python3 tools/pack_model.py "$PTE" -o "$BIN"

OCD=(openocd -f interface/stlink.cfg -f target/stm32f4x.cfg -c "transport select hla_swd")
if [ "${2:-}" != "--model-only" ]; then
    arm-none-eabi-objcopy -O binary firmware/firmware.elf "$BIN.fw"
    "${OCD[@]}" -c "program $BIN.fw 0x08000000 verify" -c "program $BIN 0x08080000 verify reset exit"
    rm -f "$BIN.fw"
else
    "${OCD[@]}" -c "program $BIN 0x08080000 verify reset exit"
fi
echo "Flashed. Run: python3 tools/send_images.py --port /dev/ttyUSB0 --val-n 500"
