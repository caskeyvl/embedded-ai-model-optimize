/* model_header.h — the 32-byte header written in front of the .pte in the
 * model Flash region (0x08080000, sectors 8-11).
 *
 *   offset  size  field
 *   0       4     magic        "L3PT" (bytes 4C 33 50 54; 0x5450334C little-endian)
 *   4       4     version      1
 *   8       4     pte_len      length of the .pte that follows, in bytes
 *   12      4     pte_crc32    CRC-32 (IEEE 802.3 / zlib.crc32) of those bytes
 *   16      16    reserved     zero
 *   32      ...   the .pte     starts 32-byte aligned, as the FlatBuffer
 *                              parser requires (>= 16)
 *
 * All fields little-endian. pack_model.py (host side) writes this format;
 * the firmware checks magic, length and CRC before it hands the .pte to
 * ExecuTorch, so a missing or corrupted model is a clear LOAD_ERR instead
 * of a fault inside the FlatBuffer parser.
 */
#pragma once
#include <stdint.h>

#define MODEL_MAGIC        0x5450334CUL   /* "L3PT" */
#define MODEL_HDR_VERSION  1U
#define MODEL_HDR_BYTES    32U

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t pte_len;
    uint32_t pte_crc32;
    uint32_t reserved[4];
} model_header_t;
