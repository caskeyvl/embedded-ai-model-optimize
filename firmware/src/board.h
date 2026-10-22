/* board.h — clock, UART (polled TX), millisecond tick, DWT cycle counter,
 * and the fault handler. Bare-metal, no HAL. */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CLOCK_HZ   168000000UL
#define APB1_HZ    42000000UL

/* USART2 (PA2 = TX, PA3 = RX, AF7), 8N1. 115200 works with every
 * USB-serial adapter; one 3,082-byte frame takes about 0.27 s at that rate. */
#ifndef UART_BAUD
#define UART_BAUD  115200UL
#endif

void board_init(void);          /* clocks, USART2 TX+RX, SysTick 1 kHz, DWT */
uint32_t board_millis(void);    /* ms since board_init (SysTick) */

static inline uint32_t dwt_cycles(void) {
    return *(volatile uint32_t *)0xE0001004UL;   /* DWT->CYCCNT */
}

void uart_putc(char c);
void uart_puts(const char *s);
void uart_put_uint(uint32_t v);
void uart_put_int(int32_t v);
void uart_put_hex(uint32_t v);  /* 0x%08x */
void uart_flush(void);          /* wait until the last byte has left */

#ifdef __cplusplus
}
#endif
