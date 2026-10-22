/* crc32.c — see crc32.h. The 1 KB table is generated at compile time by the
 * macros below, so it lands in Flash (.rodata) with no init code. */
#include "crc32.h"

#define P 0xEDB88320UL
#define S(c) (((c) & 1U) ? (((c) >> 1) ^ P) : ((c) >> 1))
#define E(n) S(S(S(S(S(S(S(S((uint32_t)(n)))))))))
#define R4(n)  E(n), E((n) + 1), E((n) + 2), E((n) + 3)
#define R16(n) R4(n), R4((n) + 4), R4((n) + 8), R4((n) + 12)
#define R64(n) R16(n), R16((n) + 16), R16((n) + 32), R16((n) + 48)

static const uint32_t crc_table[256] = { R64(0), R64(64), R64(128), R64(192) };

uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t len) {
    crc = ~crc;
    while (len--) crc = crc_table[(crc ^ *data++) & 0xFFU] ^ (crc >> 8);
    return ~crc;
}
