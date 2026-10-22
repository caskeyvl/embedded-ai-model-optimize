/*
 * stm32f407.ld — Linker script for the lab3 firmware (STM32F407VG)
 *
 * Flash is split in two (RM0090 sector map: 4x16 KB, 1x64 KB, 7x128 KB):
 *   FLASH  512 KB at 0x08000000  sectors 0-7: this firmware (code, rodata,
 *                                 the ExecuTorch runtime and kernels)
 *   MODEL  512 KB at 0x08080000  sectors 8-11: the student's model, written
 *                                 separately (32-byte header + .pte, see
 *                                 src/model_header.h). Nothing here is
 *                                 linked into it, so the same firmware ELF
 *                                 runs every submission.
 *   RAM    128 KB at 0x20000000  main SRAM (DMA-reachable): the planned
 *                                 activation arena, the frame buffer, stack
 *   CCMRAM  64 KB at 0x10000000  CPU-only: ExecuTorch's method and temp
 *                                 pools, the UART receive ring, the pixel LUT
 *
 * Based on the course's other linker scripts (lab0_hello_world_soln,
 * quantization_lab): .got before .data, .ARM.exidx, NOLOAD .ccmram, and a
 * minimum-stack ASSERT.
 */

MEMORY
{
  FLASH  (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
  MODEL  (r)   : ORIGIN = 0x08080000, LENGTH = 512K
  RAM    (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
  CCMRAM (rwx) : ORIGIN = 0x10000000, LENGTH = 64K
}

_model_region_start = ORIGIN(MODEL);
_model_region_size  = LENGTH(MODEL);

ENTRY(Reset_Handler)

SECTIONS
{
  .isr_vector :
  {
    KEEP(*(.isr_vector))
  } > FLASH

  .text :
  {
    *(.text*)
    *(.rodata*)
  } > FLASH

  /* Required by libgcc even with -fno-exceptions. */
  .ARM.extab :
  {
    *(.ARM.extab*)
    *(.gnu.linkonce.armextab.*)
  } > FLASH

  .ARM.exidx :
  {
    __exidx_start = .;
    *(.ARM.exidx*)
    *(.gnu.linkonce.armexidx.*)
    __exidx_end = .;
  } > FLASH

  .init_array :
  {
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(.init_array*))
    PROVIDE_HIDDEN(__init_array_end = .);
  } > FLASH

  .fini_array :
  {
    PROVIDE_HIDDEN(__fini_array_start = .);
    KEEP(*(.fini_array*))
    PROVIDE_HIDDEN(__fini_array_end = .);
  } > FLASH

  /* .got MUST come before .data* so the startup copy loop covers it. */
  .data :
  {
    _sdata = .;
    *(.got)
    *(.got.plt)
    *(.data*)
    _edata = .;
  } > RAM AT > FLASH

  _sidata = LOADADDR(.data);

  .bss :
  {
    _sbss = .;
    *(.bss*)
    *(COMMON)
    _ebss = .;
  } > RAM

  .ccmram (NOLOAD) :
  {
    _sccmram = .;
    *(.ccmram)
    *(.ccmram*)
    _eccmram = .;
  } > CCMRAM

  _estack = ORIGIN(RAM) + LENGTH(RAM);

  /* The arena takes most of main SRAM; fail the link rather than let the
   * stack overflow silently into .bss at runtime. */
  _min_stack = 8K;
  ASSERT(_estack - _ebss >= _min_stack, "Less than 8 KB of SRAM left for the stack")
  /* The firmware must leave sectors 8-11 to the model. */
  ASSERT(_sidata + (_edata - _sdata) <= ORIGIN(MODEL), "Firmware overlaps the model region")
}
