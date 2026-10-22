"""
pack_model.py -- wrap a .pte in the 32-byte header the firmware expects.

    python3 tools/pack_model.py submission/submission.pte -o build/model.bin

The firmware runs the model from Flash at 0x08080000 (sectors 8-11, 512 KB):

    offset  size  field
    0       4     magic      b"L3PT"
    4       4     version    1
    8       4     pte_len
    12      4     pte_crc32  zlib.crc32 of the .pte
    16      16    reserved   zero
    32      ...   the .pte

run_renode.sh and flash_board.sh call this for you; the grader does the
same. The firmware rejects a missing or corrupted model (LOAD_ERR,...)
before ExecuTorch ever parses it.
"""

import argparse
import struct
import sys
import zlib

MAGIC = b"L3PT"
VERSION = 1
HEADER_BYTES = 32
MODEL_REGION_ADDR = 0x08080000
MODEL_REGION_BYTES = 512 * 1024
MAX_PTE_BYTES = MODEL_REGION_BYTES - HEADER_BYTES     # 524,256


def pack(pte: bytes) -> bytes:
    if len(pte) > MAX_PTE_BYTES:
        raise ValueError(f".pte is {len(pte):,} bytes; the model region holds at most "
                         f"{MAX_PTE_BYTES:,} (512 KB minus the {HEADER_BYTES}-byte header)")
    hdr = MAGIC + struct.pack("<III", VERSION, len(pte), zlib.crc32(pte)) + bytes(16)
    assert len(hdr) == HEADER_BYTES
    return hdr + pte


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pte")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    with open(args.pte, "rb") as f:
        pte = f.read()
    try:
        blob = pack(pte)
    except ValueError as e:
        sys.exit(f"pack_model.py: {e}")
    with open(args.out, "wb") as f:
        f.write(blob)
    print(f"{args.out}: {len(pte):,}-byte .pte + header -> load at 0x{MODEL_REGION_ADDR:08X}")


if __name__ == "__main__":
    main()
