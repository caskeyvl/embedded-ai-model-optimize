/*
 * cortex_m_avgpool_fix.cpp — a fixed copy of ExecuTorch release/1.2's
 * backends/cortex_m/ops/op_quantized_avg_pool2d.cpp.
 *
 * The bug: the upstream kernel always calls arm_avgpool_s8() with no scratch
 * buffer (ctx.buf = nullptr). CMSIS-NN's avgpool only needs no buffer on
 * Cortex-M55/M85 (the MVE build); on a Cortex-M4 (the DSP build) it needs
 * channels * 4 bytes of int32 accumulators, and returns ARM_CMSIS_NN_ARG_ERROR
 * without them. So every int8 model with an average pool -- Model B2's GAP --
 * fails on our board with error 0x1 (Internal). The upstream backend is
 * tested on Cortex-M55 simulators, which is why nobody noticed.
 *
 * The fix: ask CMSIS-NN how much scratch it needs and allocate it from the
 * runtime's temp allocator, as the conv kernels already do.
 *
 * How it replaces the original without touching ~/executorch: this file
 * defines the same function (cortex_m::native::quantized_avg_pool2d_out).
 * The linker resolves the operator library's reference to it from this
 * object, so it never pulls the original op_quantized_avg_pool2d.cpp.obj out
 * of libcortex_m_kernels.a. The registration in cortex_m_ops_lib is unchanged.
 */

#include "cortex_m_ops_common.h"

extern "C" {
#include "arm_nnfunctions.h"
}

namespace cortex_m {
namespace native {

using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

Tensor& quantized_avg_pool2d_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Int64ArrayRef kernel_size,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const int64_t zero_point,
    const int64_t multiplier,
    const int64_t shift,
    Tensor& out) {
  constexpr int32_t activation_min = std::numeric_limits<int8_t>::min();
  constexpr int32_t activation_max = std::numeric_limits<int8_t>::max();

  const int64_t dilation_values[2] = {1, 1};
  const Int64ArrayRef dilation(dilation_values, 2);
  const bool ceil_mode = false;

  CmsisPool2DConfig pool_config;
  if (!prepare_cmsis_pool2d_config(
          context,
          "quantized_avg_pool2d_out",
          input,
          out,
          kernel_size,
          stride,
          padding,
          dilation,
          ceil_mode,
          activation_min,
          activation_max,
          pool_config)) {
    return out;
  }

  cmsis_nn_context cmsis_ctx;
  cmsis_ctx.buf = nullptr;
  cmsis_ctx.size = 0;

  /* --- The fix: give CMSIS-NN the scratch buffer it asks for. --- */
  const int32_t buffer_bytes = arm_avgpool_s8_get_buffer_size(
      pool_config.output_dims.w, pool_config.input_dims.c);
  if (buffer_bytes > 0) {
    auto buffer = context.allocate_temp(buffer_bytes, kCortexMMveAlignment);
    if (!buffer.ok()) {
      context.fail(Error::MemoryAllocationFailed);
      return out;
    }
    cmsis_ctx.buf = buffer.get();
    cmsis_ctx.size = buffer_bytes;
  }
  /* --- end of fix --- */

  const int8_t* input_data = input.const_data_ptr<int8_t>();
  int8_t* output_data = out.mutable_data_ptr<int8_t>();

  arm_cmsis_nn_status status = arm_avgpool_s8(
      &cmsis_ctx,
      &pool_config.pool_params,
      &pool_config.input_dims,
      input_data,
      &pool_config.filter_dims,
      &pool_config.output_dims,
      output_data);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    context.fail(Error::Internal);
  }

  /* Averaging int8 values that share one zero point and scale needs no
   * requantization: mean(q) - Z = mean(q - Z). */
  (void)zero_point;
  (void)multiplier;
  (void)shift;

  return out;
}

} // namespace native
} // namespace cortex_m
