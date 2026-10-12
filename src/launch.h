// launch.h - kernel launch description (grid, block, arguments) and its text format.
//
// The same text format is read by `kernc run` (CPU interpreters) and by
// harness/gpu_run.py (real GPU), and both print results in the same format,
// so outputs can be compared bit-for-bit.
//
//   kernel saxpy
//   grid 2
//   block 64
//   f32buf 4 3f800000 40000000 40400000 40800000     # pointer arg: count + hex words
//   f32 40000000                                     # scalar f32 arg (hex bits)
//   i32 00000004                                     # scalar i32 arg (hex bits)
//
// Output:
//   status ok            (or: status ub <message>)
//   buf 0 f32 4 <hex words>   (one line per pointer argument, in parameter order)
#pragma once
#include <string>
#include <vector>
#include "common.h"

namespace kernc {

struct LaunchArg {
    Type type;                    // I32, F32, PtrI32, PtrF32
    uint32_t scalar = 0;
    std::vector<uint32_t> buf;
};

struct Launch {
    std::string kernel;
    int grid = 1, block = 1;
    std::vector<LaunchArg> args;
};

Launch parseLaunch(const std::string& text);
std::string formatResult(const Launch& l, const std::string& status);

}  // namespace kernc
