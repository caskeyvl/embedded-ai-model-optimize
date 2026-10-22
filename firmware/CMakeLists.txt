# Lab 3 (model compression) firmware: one fixed ELF that runs any student's
# .pte from the model Flash region at 0x08080000. Built by the instructor;
# students and the grader only ever use the resulting firmware.elf.
#
# Build system adapted from memory_lectures/06_quantization/quantization_lab:
# ExecuTorch's Cortex-M backend (cortex_m:: ops, CMSIS-NN) for int8 layers,
# plus portable fp32 kernels for every op the baseline graph can contain, so
# layers a student leaves unquantized still run. See README.md.

cmake_minimum_required(VERSION 3.20)
project(lab3_firmware CXX C ASM)

if(NOT DEFINED EXECUTORCH_BUILD_DIR)
    message(FATAL_ERROR "EXECUTORCH_BUILD_DIR must be set. Use: cmake --preset stm32")
endif()

set(EXECUTORCH_SRC_DIR "$ENV{HOME}/executorch")
set(ET_LIB_DIR ${EXECUTORCH_BUILD_DIR})
set(CMSIS_NN_DIR "$ENV{HOME}/executorch/CMSIS-NN" CACHE PATH "CMSIS-NN source checkout")

# The planned-memory budget, in KB (src/main.cpp ARENA_KB).
set(ARENA_KB 96 CACHE STRING "Planned-memory arena budget in KB")

# ------------------------------------------------------------------ #
#  Check the ExecuTorch tree: Cortex-M backend present, and every      #
#  portable op an fp32 (unquantized) layer of the baseline can lower   #
#  to. A missing op links fine and only fails at load_method (0x14).   #
# ------------------------------------------------------------------ #
if(NOT EXISTS ${ET_LIB_DIR}/backends/cortex_m/libcortex_m_ops_lib.a)
    message(FATAL_ERROR
        "${ET_LIB_DIR} has no Cortex-M backend (backends/cortex_m/libcortex_m_ops_lib.a).\n"
        "Build the cortex-m4-cmsis tree -- see README.md.")
endif()

set(LAB3_PORTABLE_OPS
    aten::convolution.out
    aten::_native_batch_norm_legit_no_training.out
    aten::relu.out
    aten::max_pool2d_with_indices.out
    aten::mean.out
    aten::addmm.out
    aten::permute_copy.out
    aten::view_copy.out
    dim_order_ops::_clone_dim_order.out
    dim_order_ops::_to_dim_order_copy.out
)
set(ET_CACHE ${EXECUTORCH_BUILD_DIR}/CMakeCache.txt)
if(EXISTS ${ET_CACHE})
    file(STRINGS ${ET_CACHE} ET_OPS_LINE REGEX "^EXECUTORCH_SELECT_OPS_LIST:")
    string(REGEX REPLACE "^[^=]*=" "" ET_OPS "${ET_OPS_LINE}")
    string(REPLACE "," ";" ET_OPS "${ET_OPS}")
    set(ET_MISSING_OPS "")
    foreach(op ${LAB3_PORTABLE_OPS})
        if(NOT op IN_LIST ET_OPS)
            list(APPEND ET_MISSING_OPS ${op})
        endif()
    endforeach()
    if(ET_MISSING_OPS)
        string(REPLACE ";" "\n  " ET_MISSING_TEXT "${ET_MISSING_OPS}")
        message(FATAL_ERROR
            "The ExecuTorch build in ${EXECUTORCH_BUILD_DIR} is missing ops:\n"
            "  ${ET_MISSING_TEXT}\nRebuild it with the op list in README.md.")
    endif()
endif()

function(et_import name path)
    add_library(${name} STATIC IMPORTED)
    set_target_properties(${name} PROPERTIES IMPORTED_LOCATION ${ET_LIB_DIR}/${path})
endfunction()

et_import(executorch_core              libexecutorch_core.a)
et_import(executorch                   libexecutorch.a)
et_import(executorch_selected_kernels  libexecutorch_selected_kernels.a)
et_import(portable_kernels             kernels/portable/libportable_kernels.a)
et_import(kernels_util_all_deps        kernels/portable/cpu/util/libkernels_util_all_deps.a)
et_import(cortex_m_ops_lib             backends/cortex_m/libcortex_m_ops_lib.a)
et_import(cortex_m_kernels             backends/cortex_m/libcortex_m_kernels.a)
et_import(extension_data_loader        extension/data_loader/libextension_data_loader.a)
et_import(flatccrt                     third-party/flatcc_ep/lib/libflatccrt.a)

set(ET_INCLUDES
    ${EXECUTORCH_SRC_DIR}
    ${EXECUTORCH_SRC_DIR}/executorch/third-party/flatbuffers/include
    ${ET_LIB_DIR}/third-party/flatcc_ep/include
    $ENV{HOME}/executorch/et-env/lib/python3.12/site-packages/torch/include
)

set(CPU_FLAGS "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard")
set(CMAKE_C_FLAGS
    "${CPU_FLAGS} -Os -g -ffunction-sections -fdata-sections -mno-unaligned-access -DET_LOG_ENABLED=0"
    CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS
    "${CPU_FLAGS} -Os -g -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -std=c++17 -mno-unaligned-access -DET_LOG_ENABLED=0"
    CACHE STRING "" FORCE)
set(CMAKE_ASM_FLAGS "${CPU_FLAGS} -g" CACHE STRING "" FORCE)

# CMSIS-NN compiled here with -munaligned-access (the M4 handles unaligned
# LDR in hardware); everything else keeps -mno-unaligned-access for the
# FlatBuffer parser. Otherwise every 4-byte int8 load in the conv inner
# loop becomes a byte-at-a-time memcpy call: ~8x slower convs. Same
# reasoning as quantization_lab's CMakeLists.txt / README.md.
add_subdirectory(${CMSIS_NN_DIR} ${CMAKE_BINARY_DIR}/cmsis_nn EXCLUDE_FROM_ALL)
target_compile_options(cmsis-nn PRIVATE -munaligned-access)

add_executable(firmware.elf
    startup_stm32f407.s
    src/main.cpp
    src/board.c
    src/camera.c
    src/crc32.c
    src/syscalls.c
    src/cortex_m_avgpool_fix.cpp    # replaces a kernel that fails on the M4; see the file
)
target_compile_definitions(firmware.elf PRIVATE ARENA_KB=${ARENA_KB})

target_include_directories(firmware.elf PRIVATE
    ${ET_INCLUDES}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${EXECUTORCH_SRC_DIR}/executorch/backends/cortex_m/ops
    ${EXECUTORCH_SRC_DIR}/executorch/runtime/core/portable_type/c10
    ${CMSIS_NN_DIR}/Include
)

# --whole-archive: op registration happens in static initializers that
# nothing references; without it the linker drops them (0x14 at load).
target_link_libraries(firmware.elf PRIVATE
    -Wl,--whole-archive
    executorch_selected_kernels
    cortex_m_ops_lib
    -Wl,--no-whole-archive
    cortex_m_kernels
    cmsis-nn
    portable_kernels
    kernels_util_all_deps
    executorch
    executorch_core
    extension_data_loader
    flatccrt
    c m gcc
)

target_link_options(firmware.elf PRIVATE
    -nostartfiles
    -specs=nano.specs
    -T${CMAKE_SOURCE_DIR}/stm32f407.ld
    -Wl,--gc-sections
    -Wl,-Map=${CMAKE_BINARY_DIR}/firmware.map
)

add_custom_command(TARGET firmware.elf POST_BUILD
    COMMAND arm-none-eabi-objcopy -O binary firmware.elf firmware.bin
    COMMAND arm-none-eabi-size firmware.elf
    COMMENT "Generating firmware.bin; firmware size:"
)

# Flashes the firmware only (sectors 0-7). The model goes to 0x08080000
# separately: see flash_board.sh.
add_custom_target(flash
    COMMAND openocd
        -f interface/stlink.cfg
        -f target/stm32f4x.cfg
        -c "transport select hla_swd"
        -c "program ${CMAKE_BINARY_DIR}/firmware.bin 0x08000000 verify reset exit"
    DEPENDS firmware.elf
    COMMENT "Flashing firmware via OpenOCD"
    USES_TERMINAL
)
