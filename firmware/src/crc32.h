/* crc32.h — CRC-32 (IEEE 802.3, reflected, poly 0xEDB88320), the same
 * value as Python's zlib.crc32(). Table-driven, table in Flash. */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Running CRC: start with crc32_update(0, ...) and keep passing the result
 * back in, exactly like zlib.crc32(data, value). */
uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t len);

#ifdef __cplusplus
}
#endif
