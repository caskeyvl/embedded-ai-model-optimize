/* board.c — clock, UART TX, SysTick, DWT, fault reporting.
 *
 * Clock tree and USART2 setup are the course's standard ones (see
 * memory_lectures/06_quantization/quantization_lab/src/main.cpp):
 * HSE 8 MHz -> PLL -> 168 MHz SYSCLK, APB1 = 42 MHz. USART2 receive is set
 * up here too, but its interrupt handler lives in camera.c.
 */

#include "board.h"
#include "stm32f407.h"

static volatile uint32_t g_millis;

void SysTick_Handler(void) {
    g_millis++;
}

uint32_t board_millis(void) {
    return g_millis;
}

static void clock_init(void) {
    /* 1. HSE on, wait until stable. */
    RCC->CR |= RCC_CR_HSEON;
    while (!(RCC->CR & RCC_CR_HSERDY));

    /* 2. Flash: 5 wait states at 168 MHz / 3.3 V (RM0090 Table 10), plus
     * the ART accelerator's prefetch and caches. */
    FLASH->ACR = FLASH_ACR_LATENCY(5) | FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN;

    /* 3. PLL: HSE / 8 (PLLM) x 336 (PLLN) / 2 (PLLP) = 168 MHz. */
    RCC->PLLCFGR = (8U) | (336U << 6) | (0U << 16) | RCC_PLLCFGR_PLLSRC_HSE;
    RCC->CR |= RCC_CR_PLLON;
    while (!(RCC->CR & RCC_CR_PLLRDY));

    /* 4. AHB /1, APB1 /4 (42 MHz), APB2 /2 (84 MHz); switch to the PLL. */
    RCC->CFGR = RCC_CFGR_PPRE1_DIV4 | RCC_CFGR_PPRE2_DIV2 | RCC_CFGR_SW_PLL;
    while ((RCC->CFGR & RCC_CFGR_SWS_MASK) != RCC_CFGR_SWS_PLL);
}

static void uart_init(void) {
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    /* PA2 = USART2_TX, PA3 = USART2_RX, both AF7. */
    GPIOA->MODER  &= ~((0x3U << (2 * 2)) | (0x3U << (3 * 2)));
    GPIOA->MODER  |=  (GPIO_MODER_AF << (2 * 2)) | (GPIO_MODER_AF << (3 * 2));
    GPIOA->AFR[0] &= ~((0xFU << (2 * 4)) | (0xFU << (3 * 4)));
    GPIOA->AFR[0] |=  (7U << (2 * 4)) | (7U << (3 * 4));
    /* Pull-up on RX so a disconnected adapter reads as idle, not noise. */
    GPIOA->PUPDR  &= ~(0x3U << (3 * 2));
    GPIOA->PUPDR  |=  (0x1U << (3 * 2));

    RCC->APB1ENR |= RCC_APB1ENR_USART2EN;
    /* Oversampling by 16: BRR = f_APB1 / baud, rounded (115200 -> 0x16D). */
    USART2->BRR = (APB1_HZ + UART_BAUD / 2U) / UART_BAUD;
    USART2->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE | USART_CR1_RXNEIE;
    nvic_enable_irq(USART2_IRQn);
}

static void systick_init(void) {
    SysTick->LOAD = CLOCK_HZ / 1000U - 1U;
    SysTick->VAL = 0;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE | SysTick_CTRL_TICKINT | SysTick_CTRL_ENABLE;
}

static void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA;
}

void board_init(void) {
    clock_init();
    uart_init();
    systick_init();
    dwt_init();
}

/* ---- UART TX (polled) --------------------------------------------------- */

void uart_putc(char c) {
    while (!(USART2->SR & USART_SR_TXE));
    USART2->DR = (uint8_t)c;
}

void uart_puts(const char *s) {
    while (*s) uart_putc(*s++);
}

void uart_flush(void) {
    while (!(USART2->SR & USART_SR_TC));
}

void uart_put_uint(uint32_t v) {
    char tmp[10];
    int n = 0;
    do {
        tmp[n++] = (char)('0' + v % 10U);
        v /= 10U;
    } while (v);
    while (n) uart_putc(tmp[--n]);
}

void uart_put_int(int32_t v) {
    if (v < 0) {
        uart_putc('-');
        uart_put_uint((uint32_t)(-(v + 1)) + 1U);
    } else {
        uart_put_uint((uint32_t)v);
    }
}

void uart_put_hex(uint32_t v) {
    uart_puts("0x");
    for (int i = 28; i >= 0; i -= 4) uart_putc("0123456789abcdef"[(v >> i) & 0xFU]);
}

/* ---- Faults --------------------------------------------------------------
 * HardFault/MemManage/BusFault/UsageFault all land here (see the vector
 * table in startup_stm32f407.s). Prints
 *     FAULT,<CFSR>,<PC at fault>
 * and halts, so the host (send_images.py, the grader) reports the crash
 * instead of waiting for a timeout. Under GDB, the same values are in the
 * globals below. CFSR: 0x00080000 = NOCP, 0x00020000 = UNALIGNED,
 * 0x00010000 = UNDEFINSTR, 0x00000001 = IACCVIOL, 0x00008200 = precise
 * data bus error (bad address, BFAR valid). */
volatile uint32_t fault_cfsr, fault_hfsr, fault_pc, fault_lr;

void fault_report(uint32_t *frame) {
    fault_cfsr = SCB_CFSR;
    fault_hfsr = SCB_HFSR;
    fault_pc = frame[6];
    fault_lr = frame[5];
    uart_puts("\r\nFAULT,");
    uart_put_hex(fault_cfsr);
    uart_putc(',');
    uart_put_hex(fault_pc);
    uart_puts("\r\n");
    uart_flush();
    while (1);
}

__attribute__((naked)) void Fault_Handler(void) {
    __asm volatile(
        "tst lr, #4      \n"
        "ite eq          \n"
        "mrseq r0, msp   \n"
        "mrsne r0, psp   \n"
        "b fault_report  \n");
}
