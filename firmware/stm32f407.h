/* stm32f407.h — Peripheral register map for the lab3 (compression) firmware.
 *
 * RCC/FLASH/GPIO/USART2/DWT/CoreDebug fields as in the course's other
 * bare-metal projects (e.g. memory_lectures/06_quantization/quantization_lab).
 * New here: USART2 receive (RXNE interrupt), NVIC enable, SysTick, and the
 * SCB fault status registers the HardFault handler reports.
 *
 * All addresses from STM32F407 reference manual RM0090 and the Cortex-M4
 * Technical Reference Manual (SysTick/NVIC/SCB/DWT/CoreDebug are core
 * peripherals, not ST-specific).
 */

#ifndef STM32F407_H
#define STM32F407_H

#include <stdint.h>

/* ── Base addresses ──────────────────────────────────────────────────────── */
#define PERIPH_BASE     0x40000000UL
#define APB1_BASE       (PERIPH_BASE + 0x00000000UL)
#define AHB1_BASE       (PERIPH_BASE + 0x00020000UL)

#define GPIOA_BASE      (AHB1_BASE + 0x0000UL)
#define RCC_BASE        (AHB1_BASE + 0x3800UL)
#define USART2_BASE     (APB1_BASE + 0x4400UL)
#define FLASH_BASE      0x40023C00UL

#define SysTick_BASE    0xE000E010UL
#define NVIC_ISER_BASE  0xE000E100UL
#define SCB_CFSR_ADDR   0xE000ED28UL
#define SCB_HFSR_ADDR   0xE000ED2CUL
#define SCB_MMFAR_ADDR  0xE000ED34UL
#define SCB_BFAR_ADDR   0xE000ED38UL
#define DWT_BASE        0xE0001000UL
#define CoreDebug_BASE  0xE000EDF0UL

/* ── GPIO ────────────────────────────────────────────────────────────────── */
typedef struct {
    volatile uint32_t MODER;
    volatile uint32_t OTYPER;
    volatile uint32_t OSPEEDR;
    volatile uint32_t PUPDR;
    volatile uint32_t IDR;
    volatile uint32_t ODR;
    volatile uint32_t BSRR;
    volatile uint32_t LCKR;
    volatile uint32_t AFR[2];
} GPIO_t;

#define GPIOA   ((GPIO_t *)GPIOA_BASE)
#define GPIO_MODER_AF       0x2U

/* ── RCC ─────────────────────────────────────────────────────────────────── */
typedef struct {
    volatile uint32_t CR;           /* 0x00 */
    volatile uint32_t PLLCFGR;      /* 0x04 */
    volatile uint32_t CFGR;         /* 0x08 */
    volatile uint32_t CIR;          /* 0x0C */
    volatile uint32_t AHB1RSTR;     /* 0x10 */
    volatile uint32_t AHB2RSTR;     /* 0x14 */
    volatile uint32_t AHB3RSTR;     /* 0x18 */
    uint32_t          RESERVED0;    /* 0x1C */
    volatile uint32_t APB1RSTR;     /* 0x20 */
    volatile uint32_t APB2RSTR;     /* 0x24 */
    uint32_t          RESERVED1[2]; /* 0x28-0x2C */
    volatile uint32_t AHB1ENR;      /* 0x30 */
    volatile uint32_t AHB2ENR;      /* 0x34 */
    volatile uint32_t AHB3ENR;      /* 0x38 */
    uint32_t          RESERVED2;    /* 0x3C */
    volatile uint32_t APB1ENR;      /* 0x40 */
} RCC_t;

#define RCC     ((RCC_t *)RCC_BASE)

#define RCC_CR_HSEON        (1U << 16)
#define RCC_CR_HSERDY       (1U << 17)
#define RCC_CR_PLLON        (1U << 24)
#define RCC_CR_PLLRDY       (1U << 25)
#define RCC_PLLCFGR_PLLSRC_HSE  (1U << 22)
#define RCC_CFGR_SW_PLL     (0x2U << 0)
#define RCC_CFGR_SWS_PLL    (0x2U << 2)
#define RCC_CFGR_SWS_MASK   (0x3U << 2)
#define RCC_CFGR_PPRE1_DIV4 (0x5U << 10)  /* APB1 = AHB/4 */
#define RCC_CFGR_PPRE2_DIV2 (0x4U << 13)  /* APB2 = AHB/2 */
#define RCC_AHB1ENR_GPIOAEN (1U << 0)
#define RCC_APB1ENR_USART2EN  (1U << 17)

/* ── FLASH interface ─────────────────────────────────────────────────────── */
typedef struct {
    volatile uint32_t ACR;
} FLASH_t;

#define FLASH   ((FLASH_t *)FLASH_BASE)
#define FLASH_ACR_LATENCY(n)    ((n) & 0x7U)
#define FLASH_ACR_PRFTEN        (1U << 8)
#define FLASH_ACR_ICEN          (1U << 9)
#define FLASH_ACR_DCEN          (1U << 10)

/* ── USART ───────────────────────────────────────────────────────────────── */
typedef struct {
    volatile uint32_t SR;
    volatile uint32_t DR;
    volatile uint32_t BRR;
    volatile uint32_t CR1;
    volatile uint32_t CR2;
    volatile uint32_t CR3;
    volatile uint32_t GTPR;
} USART_t;

#define USART2  ((USART_t *)USART2_BASE)

#define USART_SR_ORE    (1U << 3)
#define USART_SR_RXNE   (1U << 5)
#define USART_SR_TC     (1U << 6)
#define USART_SR_TXE    (1U << 7)
#define USART_CR1_RE    (1U << 2)
#define USART_CR1_TE    (1U << 3)
#define USART_CR1_RXNEIE (1U << 5)
#define USART_CR1_UE    (1U << 13)

#define USART2_IRQn     38

/* ── NVIC ────────────────────────────────────────────────────────────────── */
#define NVIC_ISER       ((volatile uint32_t *)NVIC_ISER_BASE)
static inline void nvic_enable_irq(int irqn) {
    NVIC_ISER[irqn >> 5] = 1U << (irqn & 31);
}

/* ── SysTick ─────────────────────────────────────────────────────────────── */
typedef struct {
    volatile uint32_t CTRL;
    volatile uint32_t LOAD;
    volatile uint32_t VAL;
    volatile uint32_t CALIB;
} SysTick_t;

#define SysTick ((SysTick_t *)SysTick_BASE)
#define SysTick_CTRL_ENABLE     (1U << 0)
#define SysTick_CTRL_TICKINT    (1U << 1)
#define SysTick_CTRL_CLKSOURCE  (1U << 2)   /* 1 = processor clock */

/* ── SCB fault status ────────────────────────────────────────────────────── */
#define SCB_CFSR    (*(volatile uint32_t *)SCB_CFSR_ADDR)
#define SCB_HFSR    (*(volatile uint32_t *)SCB_HFSR_ADDR)

/* ── DWT cycle counter ───────────────────────────────────────────────────── */
typedef struct {
    volatile uint32_t CTRL;    /* 0x00 */
    volatile uint32_t CYCCNT;  /* 0x04 */
} DWT_t;

#define DWT     ((DWT_t *)DWT_BASE)
#define DWT_CTRL_CYCCNTENA  (1U << 0)

/* CoreDebug->DEMCR must have TRCENA set before DWT->CTRL has any effect. */
typedef struct {
    uint32_t          RESERVED0[3]; /* DHCSR, DCRSR, DCRDR -- unused here */
    volatile uint32_t DEMCR;        /* 0xE000EDFC */
} CoreDebug_t;

#define CoreDebug   ((CoreDebug_t *)CoreDebug_BASE)
#define CoreDebug_DEMCR_TRCENA  (1U << 24)

#endif /* STM32F407_H */
