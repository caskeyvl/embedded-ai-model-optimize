/*
 * main.cpp — Lab 3 (model compression) firmware for the STM32F4DISCOVERY.
 *
 * The same ELF runs every student's model. The model is not linked in: it
 * is written separately to the model Flash region (0x08080000, sectors
 * 8-11) as a 32-byte header plus the .pte (src/model_header.h), and run
 * from Flash in place. The firmware:
 *
 *   1. checks the header (magic, length, CRC-32) and loads the program;
 *   2. checks the budgets: the program's planned memory must fit the fixed
 *      ARENA_BYTES arena, and its input must be a float [1,3,32,32] tensor;
 *   3. loads the method (every operator must be registered: CMSIS-NN
 *      cortex_m:: ops for int8, portable aten:: ops for fp32 fallbacks);
 *   4. loops: receive a frame from the "camera" (USART2, src/camera.h),
 *      normalize it straight into the model's input tensor, execute, and
 *      report the predicted class.
 *
 * UART protocol, device -> host, one ASCII line each (\r\n):
 *
 *   BOOT,<fw_version>,<pte_len>,<planned_arena_bytes>
 *   INFO,<key>,<value>...                     (informational; ignore freely)
 *   READY                                     ready for frames
 *   LOAD_ERR,<reason>[,<detail>]              model rejected; stays alive
 *   R,<idx>,<pred_class>,<cycles>,<status>    one per frame received
 *   FAULT,<cfsr>,<pc>                         hard fault; halted (board.c)
 *
 *   status: OK | CRC_ERR | LEN_ERR | TIMEOUT | EXEC_ERR:<hex>
 *   cycles: DWT cycles for method.execute() only. Meaningful on the real
 *           board; under Renode it is not cycle-accurate.
 *
 * Memory (see README.md "Memory map"):
 *   planned_pool  main SRAM  ARENA_BYTES  the activation arena, incl. the
 *                                          fp32 input tensor (12 KB)
 *   frame         main SRAM  3 KB          camera frame buffer (DMA-able)
 *   method_pool   CCM        32 KB         ExecuTorch per-method state
 *   temp_pool     CCM        16 KB         kernel scratch (CMSIS-NN im2col)
 *   rx ring, LUT  CCM        4 KB + 3 KB
 *   the model     Flash      read in place at 0x08080020
 */

#include <cstdint>
#include <cstring>

#include "board.h"
#include "camera.h"
#include "crc32.h"
#include "model_header.h"

#include "executorch/runtime/executor/program.h"
#include "executorch/runtime/executor/method.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/core/hierarchical_allocator.h"
#include "executorch/runtime/platform/runtime.h"
#include "executorch/extension/data_loader/buffer_data_loader.h"

using namespace executorch::runtime;
using namespace executorch::extension;

#define FW_VERSION "lab3-fw-1.0"

/* The planned-memory budget (README.md "Budgets"). Every submission's
 * program must plan at most this much: activations and the fp32 input. */
#ifndef ARENA_KB
#define ARENA_KB 96
#endif
#define ARENA_BYTES (ARENA_KB * 1024U)

#define NUM_CLASSES 10
/* Partial frame: give up after this long without a byte. */
#define FRAME_TIMEOUT_MS 2000U

/* CIFAR-10 normalization, identical to data.py (MEAN, STD). */
static const float MEAN[3] = {0.4914f, 0.4822f, 0.4465f};
static const float STD[3]  = {0.2470f, 0.2435f, 0.2616f};

extern "C" const uint8_t _model_region_start[];
extern "C" const uint8_t _model_region_size[];   /* address == size */

/* ---- Memory -------------------------------------------------------------- */

static uint8_t planned_pool[ARENA_BYTES] __attribute__((aligned(16)));
static uint8_t frame[CAMERA_FRAME_BYTES] __attribute__((aligned(4)));

__attribute__((section(".ccmram"), aligned(16))) static uint8_t method_pool[32 * 1024];
__attribute__((section(".ccmram"), aligned(16))) static uint8_t temp_pool[16 * 1024];
__attribute__((section(".ccmram"))) static float pixel_lut[3][256];

/* Temp allocator that records its high-water mark (reported as INFO). */
class PeakAllocator : public MemoryAllocator {
public:
    PeakAllocator(uint32_t size, uint8_t *base) : MemoryAllocator(size, base) {}
    void *allocate(size_t size, size_t alignment = kDefaultAlignment) override {
        uint8_t *p = (uint8_t *)MemoryAllocator::allocate(size, alignment);
        if (p != nullptr && (uint32_t)(p + size - base_address()) > peak) {
            peak = (uint32_t)(p + size - base_address());
        }
        return p;
    }
    uint32_t peak = 0;
};

/* Exposed for GDB: the last prediction and per-frame counters. */
volatile int32_t last_pred = -1;
volatile uint32_t last_cycles = 0;
volatile uint32_t frames_ok = 0, frames_bad = 0;

/* ---- Output helpers -------------------------------------------------------- */

static void line_end(void) {
    uart_puts("\r\n");
}

/* LOAD_ERR,<reason>[,<detail>] then idle forever: the host sees the reason
 * instead of a silent hang. Frames sent anyway are drained and answered
 * so a host that ignores LOAD_ERR still gets one line per frame. */
static void load_error(const char *reason, int has_detail, uint32_t detail) {
    uart_puts("LOAD_ERR,");
    uart_puts(reason);
    if (has_detail) {
        uart_putc(',');
        uart_put_hex(detail);
    }
    line_end();
    while (1) {
        int st = camera_capture(frame, CAMERA_FRAME_BYTES, 1000);
        if (st == CAM_NO_FRAME) continue;
        uart_puts("R,");
        uart_put_uint(camera_frame_index());
        uart_puts(",-1,0,LOAD_ERR");
        line_end();
    }
}

static void report(uint16_t idx, int32_t pred, uint32_t cycles, const char *status,
                   int has_code, uint32_t code) {
    uart_puts("R,");
    uart_put_uint(idx);
    uart_putc(',');
    uart_put_int(pred);
    uart_putc(',');
    uart_put_uint(cycles);
    uart_putc(',');
    uart_puts(status);
    if (has_code) {
        uart_putc(':');
        uart_put_hex(code);
    }
    line_end();
}

/* ---- Model region -------------------------------------------------------- */

/* Returns the .pte (and its length) if the header is valid, else reports
 * LOAD_ERR. */
static const uint8_t *check_model_region(uint32_t *pte_len) {
    const model_header_t *hdr = (const model_header_t *)_model_region_start;
    const uint32_t region = (uint32_t)_model_region_size;
    *pte_len = 0;
    if (hdr->magic != MODEL_MAGIC) {
        /* Erased Flash reads 0xFFFFFFFF: no model was ever written. */
        load_error(hdr->magic == 0xFFFFFFFFUL ? "NO_MODEL" : "BAD_MAGIC", 1, hdr->magic);
    }
    if (hdr->version != MODEL_HDR_VERSION) load_error("BAD_HEADER_VERSION", 1, hdr->version);
    if (hdr->pte_len == 0 || hdr->pte_len > region - MODEL_HDR_BYTES)
        load_error("BAD_LENGTH", 1, hdr->pte_len);
    const uint8_t *pte = _model_region_start + MODEL_HDR_BYTES;
    uint32_t crc = crc32_update(0, pte, hdr->pte_len);
    if (crc != hdr->pte_crc32) load_error("BAD_CRC", 1, crc);
    *pte_len = hdr->pte_len;
    return pte;
}

/* ---- Preprocessing -------------------------------------------------------- */

/* pixel_lut[c][p] = (p / 255 - MEAN[c]) / STD[c], the same float32
 * operations as data.py normalize(), so the board's input tensor is
 * bit-for-bit the one PyTorch computes. */
static void build_lut(void) {
    for (int c = 0; c < 3; c++)
        for (int p = 0; p < 256; p++)
            pixel_lut[c][p] = ((float)p / 255.0f - MEAN[c]) / STD[c];
}

/* frame (uint8 HWC) -> input tensor (float, whatever dim order the program
 * recorded: NHWC for the int8 export, NCHW otherwise), through the tensor's
 * own strides. */
static void preprocess(const uint8_t *src, float *dst, const int32_t *stride) {
    for (int h = 0; h < CAMERA_H; h++)
        for (int w = 0; w < CAMERA_W; w++)
            for (int c = 0; c < CAMERA_C; c++)
                dst[c * stride[1] + h * stride[2] + w * stride[3]] = pixel_lut[c][*src++];
}

static int32_t argmax(const float *p, int n) {
    int32_t best = 0;
    for (int i = 1; i < n; i++)
        if (p[i] > p[best]) best = i;
    return best;
}

/* -------------------------------------------------------------------------- */

int main(void) {
    board_init();
    camera_init();
    runtime_init();
    build_lut();

    uart_puts("\r\n");
    uint32_t pte_len = 0;
    const uint8_t *pte = nullptr;
    uint32_t planned = 0;

    /* BOOT is printed once the model's size and plan are known, or with
     * zeros just before a LOAD_ERR. */
    struct Boot {
        static void print(uint32_t len, uint32_t plan) {
            uart_puts("BOOT," FW_VERSION ",");
            uart_put_uint(len);
            uart_putc(',');
            uart_put_uint(plan);
            line_end();
        }
    };

    const model_header_t *hdr = (const model_header_t *)_model_region_start;
    if (hdr->magic != MODEL_MAGIC) Boot::print(0, 0);
    pte = check_model_region(&pte_len);

    BufferDataLoader loader(pte, pte_len);
    Result<Program> program_result = Program::load(&loader, Program::Verification::Minimal);
    if (!program_result.ok()) {
        Boot::print(pte_len, 0);
        load_error("PROGRAM", 1, (uint32_t)program_result.error());
    }
    Program program = std::move(program_result.get());

    Result<MethodMeta> meta = program.method_meta("forward");
    if (!meta.ok()) {
        Boot::print(pte_len, 0);
        load_error("METHOD_META", 1, (uint32_t)meta.error());
    }
    for (size_t i = 0; i < meta->num_memory_planned_buffers(); i++)
        planned += (uint32_t)meta->memory_planned_buffer_size(i).get();
    Boot::print(pte_len, planned);

    uart_puts("INFO,arena,");
    uart_put_uint(planned);
    uart_putc(',');
    uart_put_uint(ARENA_BYTES);
    line_end();
    if (meta->num_memory_planned_buffers() != 1)
        load_error("PLANNED_BUFFERS", 1, (uint32_t)meta->num_memory_planned_buffers());
    if (planned > ARENA_BYTES) load_error("ARENA_TOO_SMALL", 1, planned);

    /* The input must be the float [1,3,32,32] tensor export.py produces. */
    if (meta->num_inputs() != 1 || meta->num_outputs() != 1)
        load_error("IO_COUNT", 1, (uint32_t)(meta->num_inputs() * 16 + meta->num_outputs()));
    Result<TensorInfo> in_meta = meta->input_tensor_meta(0);
    if (!in_meta.ok()) load_error("INPUT_META", 1, (uint32_t)in_meta.error());
    {
        auto s = in_meta->sizes();
        if (in_meta->scalar_type() != executorch::aten::ScalarType::Float || s.size() != 4 ||
            s[0] != 1 || s[1] != 3 || s[2] != CAMERA_H || s[3] != CAMERA_W)
            load_error("INPUT_NOT_FLOAT_1x3x32x32", 1, (uint32_t)in_meta->scalar_type());
    }

    MemoryAllocator method_allocator(sizeof(method_pool), method_pool);
    PeakAllocator temp_allocator(sizeof(temp_pool), temp_pool);
    Span<uint8_t> planned_span(planned_pool, planned);
    HierarchicalAllocator planned_allocator({&planned_span, 1});
    MemoryManager memory_manager(&method_allocator, &planned_allocator, &temp_allocator);

    Result<Method> method_result = program.load_method("forward", &memory_manager);
    if (!method_result.ok()) {
        /* 0x14 OperatorMissing: an op the firmware has no kernel for.
         * 0x21 MemoryAllocationFailed: method_pool too small. */
        load_error("LOAD_METHOD", 1, (uint32_t)method_result.error());
    }
    Method method = std::move(method_result.get());
    uint8_t *method_pool_end = (uint8_t *)method_allocator.allocate(1, 1);
    uart_puts("INFO,method_pool,");
    uart_put_uint((uint32_t)(method_pool_end - method_pool));
    uart_putc(',');
    uart_put_uint(sizeof(method_pool));
    line_end();

    /* The input tensor lives in the planned arena (export.py plans graph
     * inputs). Write each frame into it directly: no 12 KB staging buffer.
     * mutable_input() is deprecated upstream in favour of set_input(), which
     * would need that staging copy. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    executorch::aten::Tensor in_tensor = method.mutable_input(0).toTensor();
#pragma GCC diagnostic pop
    float *in_data = in_tensor.mutable_data_ptr<float>();
    if (in_data == nullptr) load_error("INPUT_NOT_PLANNED", 0, 0);
    int32_t stride[4];
    for (int d = 0; d < 4; d++) stride[d] = (int32_t)in_tensor.strides()[d];
    uart_puts("INFO,input_layout,");
    uart_puts(stride[1] == 1 ? "NHWC" : "NCHW");
    line_end();

    uart_puts("READY");
    line_end();

    bool reported_temp = false;
    for (;;) {
        int st = camera_capture(frame, CAMERA_FRAME_BYTES, FRAME_TIMEOUT_MS);
        if (st == CAM_NO_FRAME) continue;
        uint16_t idx = camera_frame_index();
        if (st != CAM_OK) {
            frames_bad++;
            report(idx, -1, 0,
                   st == CAM_CRC_ERR ? "CRC_ERR" : st == CAM_LEN_ERR ? "LEN_ERR" : "TIMEOUT", 0, 0);
            continue;
        }

        preprocess(frame, in_data, stride);
        uint32_t t0 = dwt_cycles();
        Error err = method.execute();
        uint32_t cycles = dwt_cycles() - t0;
        if (err != Error::Ok) {
            frames_bad++;
            /* execute() has already reset itself to instruction 0. */
            report(idx, -1, cycles, "EXEC_ERR", 1, (uint32_t)err);
            continue;
        }

        const EValue &out = method.get_output(0);
        if (!out.isTensor() || out.toTensor().scalar_type() != executorch::aten::ScalarType::Float ||
            out.toTensor().numel() != NUM_CLASSES) {
            frames_bad++;
            report(idx, -1, cycles, "EXEC_ERR", 1, 0xBAD0u);
            continue;
        }
        last_pred = argmax(out.toTensor().const_data_ptr<float>(), NUM_CLASSES);
        last_cycles = cycles;
        frames_ok++;
        report(idx, last_pred, cycles, "OK", 0, 0);

        if (!reported_temp) {
            uart_puts("INFO,temp_pool_peak,");
            uart_put_uint(temp_allocator.peak);
            uart_putc(',');
            uart_put_uint(sizeof(temp_pool));
            line_end();
            reported_temp = true;
        }
    }
}
