# Lab 3 firmware

One fixed firmware for every submission. It runs a CIFAR-10 `.pte` from its
own Flash region and classifies RGB frames that arrive over USART2 (the
"camera"). Students and the grader use the prebuilt `firmware.elf`; only the
instructor builds it. In the student repo this directory holds a read-only
copy of the source, for reference.

Bare metal (no HAL), ExecuTorch 1.2 with the Cortex-M backend (CMSIS-NN
v7.0.0) for int8 layers, plus portable fp32 kernels for every op an
unquantized layer of the baseline can lower to.

## Boot and main loop

1. Clock (HSE → PLL → 168 MHz), USART2 (PA2 TX / PA3 RX, AF7, 115200 8N1,
   RX interrupt), SysTick (1 kHz), DWT cycle counter.
2. Check the model region at 0x0808_0000: the 32-byte header's magic,
   version, length, and the CRC-32 of the `.pte` (`src/model_header.h`).
3. `Program::load` straight from Flash (no copy). Read the planned-memory
   size from the program and check it against the arena (`ARENA_KB`, 96).
   Check that the input is float `[1,3,32,32]`.
4. `load_method` with the method pool and temp pool in CCM and the arena in
   main SRAM.
5. Print `READY`, then loop: receive a frame, normalize it straight into the
   input tensor's slot in the arena (per-channel LUT of
   `(p/255 − mean)/std` in float32, the same ops as `data.normalize`, in
   whatever dim order the program recorded; NHWC for `export.py`'s output),
   `method.execute()` timed with DWT, argmax, print the result.

## UART protocol

Host → device, one frame per image (little-endian):

```
A5 5A | idx:uint16 | len:uint16 = 3072 | payload: 32×32×3 uint8, RGB, HWC | crc32:uint32
```

`crc32` is `zlib.crc32(idx|len|payload)`. Frames are parsed out of a 4 KB
receive ring, which is bigger than one frame. The host sends a frame, waits
for its `R` line, then sends the next.

Device → host, one ASCII line each:

| Line | When |
|---|---|
| `BOOT,lab3-fw-1.0,<pte_len>,<planned_bytes>` | once, at boot |
| `INFO,arena,<planned>,<arena>` / `INFO,method_pool,<used>,<size>` / `INFO,input_layout,NHWC` / `INFO,temp_pool_peak,<peak>,<size>` | informational |
| `READY` | model loaded; send frames |
| `R,<idx>,<pred>,<cycles>,<status>` | one per frame; status `OK`, `CRC_ERR`, `LEN_ERR`, `TIMEOUT`, `EXEC_ERR:<hex>` (`pred` = −1 unless OK) |
| `LOAD_ERR,<reason>[,<hex>]` | model rejected; the firmware keeps answering frames with `R,<idx>,-1,0,LOAD_ERR` |
| `FAULT,<CFSR>,<PC>` | HardFault/MemManage/BusFault/UsageFault; halted |

`LOAD_ERR` reasons: `NO_MODEL` (erased Flash), `BAD_MAGIC`,
`BAD_HEADER_VERSION`, `BAD_LENGTH`, `BAD_CRC`, `PROGRAM`, `METHOD_META`,
`PLANNED_BUFFERS`, `ARENA_TOO_SMALL`, `IO_COUNT`, `INPUT_META`,
`INPUT_NOT_FLOAT_1x3x32x32`, `INPUT_NOT_PLANNED`, `LOAD_METHOD` (0x14 =
OperatorMissing, 0x21 = MemoryAllocationFailed).

`cycles` counts `method.execute()` only. Renode has no DWT model (the
platform maps a RAM stub there), so it reads 0 under Renode. Only the real
board gives latency.

## Memory map

| Region | Contents | Size |
|---|---|---:|
| Flash 0x0800_0000 (sectors 0–7) | code + rodata, ExecuTorch runtime, CMSIS-NN, portable kernels | 225,861 B of 512 KB |
| Flash 0x0808_0000 (sectors 8–11) | model header + `.pte` | ≤ 512 KB |
| SRAM | `planned_pool` (arena) | 96 KB |
| SRAM | `frame` (DMA-able frame buffer) | 3 KB |
| SRAM | the rest of `.data`/`.bss` (kernel registry, runtime state) | 3.6 KB |
| SRAM | stack (the link fails below 8 KB) | 25.5 KB |
| CCM | `method_pool` (ExecuTorch method state; ~7 KB used for 13 instructions) | 32 KB |
| CCM | `temp_pool` (CMSIS-NN im2col scratch, reset per kernel; ≤ 9 KB for a 256-ch conv) | 16 KB |
| CCM | UART receive ring, pixel LUT | 4 KB + 3 KB |
| CCM | total | 56,320 B of 64 KB |

## Deviations from the build spec

- **Interrupt-driven RX instead of DMA.** Renode's STM32 UART model doesn't
  reliably support DMA requests. The protocol is request/response, so
  per-byte interrupts at 115200 baud cost nothing measurable. The frame
  buffer is still in main SRAM, so a DMA driver is a drop-in change in
  `src/camera.c`.
- **No test mode.** Renode spends its time in inference, not in UART bytes
  (~3 KB per frame against ~100 M instructions per inference), so the
  grader streams frames over the socket like the board does, and runs
  several Renode instances in parallel instead.
- **`FAULT` carries the PC as well** (`FAULT,<cfsr>,<pc>`).

## Building (instructor)

Needs `arm-none-eabi-gcc` 13, the ExecuTorch tree
`~/executorch/executorch-build/cortex-m4-cmsis` built with
`EXECUTORCH_BUILD_CORTEX_M=ON` (the `quantization_lab/` tree; its op
list is a superset of what `CMakeLists.txt` checks for), and the CMSIS-NN
v7.0.0 checkout at `~/executorch/CMSIS-NN`.

```bash
source ~/executorch/et-env/bin/activate
cmake --preset stm32 && cmake --build --preset stm32     # -> build/stm32/firmware.elf
../solution/sync_release.sh                              # copy it to provided/ and test/
```

`ARENA_KB` is a CMake cache variable (`cmake --preset stm32 -DARENA_KB=96`).
If you change it, also change `ARENA_BUDGET_BYTES` in `export.py`,
`check_submission.py` and `budget.py`.

Source files: `src/main.cpp` (load, checks, loop), `src/camera.{c,h}`
(frame protocol, RX ISR), `src/board.{c,h}` (clock, UART TX, SysTick, DWT,
fault handler), `src/crc32.{c,h}`, `src/model_header.h`, `src/syscalls.c`
(newlib stubs, byte-wise mem*), and `src/cortex_m_avgpool_fix.cpp`.
ExecuTorch 1.2's `quantized_avg_pool2d` calls CMSIS-NN without the scratch
buffer the M4 (DSP) build needs and fails with error 0x1; this is the
`quantization_lab/`'s fix, which the linker uses instead of the
original.
