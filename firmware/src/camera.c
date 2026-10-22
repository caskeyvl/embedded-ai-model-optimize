/* camera.c — see camera.h for the frame format. */
#include "camera.h"

#include "board.h"
#include "crc32.h"
#include "stm32f407.h"

/* 4 KB ring: more than one whole frame (3,082 bytes), so even a host that
 * sends the next frame before reading the previous result can't overrun
 * it. CCM is fine: only the CPU (the ISR) writes it. */
#define RING_BYTES 4096U
__attribute__((section(".ccmram"))) static uint8_t ring[RING_BYTES];
static volatile uint32_t ring_head;   /* written by the ISR */
static uint32_t ring_tail;            /* read by camera_capture() */
static volatile uint32_t overruns;
static uint16_t last_idx;

void USART2_IRQHandler(void) {
    uint32_t sr = USART2->SR;
    if (sr & (USART_SR_RXNE | USART_SR_ORE)) {
        uint8_t b = (uint8_t)USART2->DR;   /* reading DR clears RXNE and ORE */
        if (sr & USART_SR_ORE) overruns++;
        uint32_t h = ring_head;
        if (h - ring_tail < RING_BYTES) {
            ring[h % RING_BYTES] = b;
            ring_head = h + 1U;
        } else {
            overruns++;
        }
    }
}

int camera_init(void) {
    ring_head = ring_tail = 0;
    overruns = 0;
    return 0;
}

uint32_t camera_overruns(void) {
    return overruns;
}

uint16_t camera_frame_index(void) {
    return last_idx;
}

/* Next byte from the ring, waiting up to timeout_ms. */
static int get_byte(uint8_t *b, uint32_t timeout_ms) {
    uint32_t t0 = board_millis();
    while (ring_head == ring_tail) {
        if (board_millis() - t0 > timeout_ms) return 0;
    }
    *b = ring[ring_tail % RING_BYTES];
    ring_tail++;
    return 1;
}

static int get_u16(uint16_t *v, uint8_t *raw, uint32_t timeout_ms) {
    if (!get_byte(&raw[0], timeout_ms) || !get_byte(&raw[1], timeout_ms)) return 0;
    *v = (uint16_t)(raw[0] | (raw[1] << 8));
    return 1;
}

int camera_capture(uint8_t *frame, size_t len, uint32_t timeout_ms) {
    uint8_t b;

    /* Hunt for SYNC (0xA5 0x5A). Anything else is skipped. */
    if (!get_byte(&b, timeout_ms)) return CAM_NO_FRAME;
    for (;;) {
        if (b == 0xA5) {
            if (!get_byte(&b, timeout_ms)) return CAM_TIMEOUT;
            if (b == 0x5A) break;
            continue;            /* b may itself be 0xA5: re-test it */
        }
        if (!get_byte(&b, timeout_ms)) return CAM_NO_FRAME;
    }

    uint8_t hdr[4];
    uint16_t idx, flen;
    if (!get_u16(&idx, &hdr[0], timeout_ms)) return CAM_TIMEOUT;
    last_idx = idx;
    if (!get_u16(&flen, &hdr[2], timeout_ms)) return CAM_TIMEOUT;
    if (flen != len) return CAM_LEN_ERR;   /* resync on the next SYNC */

    uint32_t crc = crc32_update(0, hdr, 4);
    for (size_t i = 0; i < len; i++) {
        if (!get_byte(&frame[i], timeout_ms)) return CAM_TIMEOUT;
    }
    crc = crc32_update(crc, frame, len);

    uint32_t rx_crc = 0;
    for (int i = 0; i < 4; i++) {
        if (!get_byte(&b, timeout_ms)) return CAM_TIMEOUT;
        rx_crc |= (uint32_t)b << (8 * i);
    }
    return rx_crc == crc ? CAM_OK : CAM_CRC_ERR;
}
