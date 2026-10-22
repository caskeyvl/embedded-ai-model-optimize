/* camera.h — the "camera": RGB888 frames that arrive over USART2.
 *
 * Host -> device, one frame per image (all fields little-endian):
 *
 *     SYNC 0xA5 0x5A | idx:uint16 | len:uint16 (= 3072) | payload[len] | crc32:uint32
 *
 * payload is the image as uint8 RGB, HWC (row-major, 3 bytes per pixel),
 * 32 x 32 x 3 = 3072 bytes. crc32 is zlib.crc32() of the 4 bytes idx|len
 * followed by the payload (everything between SYNC and the CRC).
 *
 * Reception is interrupt-driven (RXNE) into a ring buffer in CCM RAM;
 * camera_capture() parses frames out of the ring into the caller's frame
 * buffer, which lives in main SRAM so a DMA-based driver could fill it the
 * same way.
 */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CAMERA_W        32
#define CAMERA_H        32
#define CAMERA_C        3
#define CAMERA_FRAME_BYTES  (CAMERA_W * CAMERA_H * CAMERA_C)   /* 3072 */

enum {
    CAM_OK = 0,        /* a complete frame with a good CRC is in frame[] */
    CAM_NO_FRAME = 1,  /* nothing arrived within timeout_ms (not an error) */
    CAM_TIMEOUT = 2,   /* a frame started, then stalled for > timeout_ms */
    CAM_CRC_ERR = 3,   /* complete frame, CRC mismatch */
    CAM_LEN_ERR = 4,   /* header's len != the expected frame size */
};

int camera_init(void);

/* Blocks until one frame arrives or timeout (no byte for timeout_ms).
 * Returns one of the CAM_* codes; 0 on success. */
int camera_capture(uint8_t *frame, size_t len, uint32_t timeout_ms);

/* The idx field of the frame camera_capture() last parsed (valid for every
 * return code except CAM_NO_FRAME, and CAM_TIMEOUT before the header). */
uint16_t camera_frame_index(void);

/* Receive-ring overruns since boot (bytes dropped because the ring was
 * full, plus USART hardware overruns). Nonzero means the host is sending
 * faster than the protocol allows. */
uint32_t camera_overruns(void);

#ifdef __cplusplus
}
#endif
