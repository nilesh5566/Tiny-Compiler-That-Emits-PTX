# Optimiser benchmark

Static counts: PTX excludes the parameter-loading prologue, identical at every level.

| kernel | IR -O0 | IR -O1 | PTX -O0 | PTX -O1 | PTX -O1 --fma | PTX reduction |
|---|---:|---:|---:|---:|---:|---:|
| collatz.kern | 43 | 41 | 43 | 41 | 41 | 5% |
| histogram_rows.kern | 68 | 58 | 67 | 57 | 57 | 15% |
| mandelbrot.kern | 65 | 54 | 66 | 55 | 53 | 17% |
| polynomial.kern | 33 | 32 | 34 | 33 | 32 | 3% |
| relu.kern | 37 | 33 | 38 | 34 | 34 | 11% |
| saxpy.kern | 17 | 13 | 19 | 15 | 14 | 21% |
| sum_rows.kern | 27 | 24 | 28 | 25 | 25 | 11% |
| **total** | **290** | **255** | **295** | **260** | **256** | **12%** |

Executed IR instructions (all threads, CPU IR interpreter):

| test | -O0 | -O1 | reduction |
|---|---:|---:|---:|
| collatz.kern | 94,548 | 69,391 | 27% |
| common_subexpr.kern | 304 | 240 | 21% |
| const_fold_semantics.kern | 65 | 46 | 29% |
| copy_prop.kern | 92 | 64 | 30% |
| copy_prop_param.kern | 160 | 138 | 14% |
| grid_geometry.kern | 270 | 270 | 0% |
| histogram_rows.kern | 19,625 | 13,718 | 30% |
| le_vs_lt.kern | 224 | 200 | 11% |
| mandelbrot.kern | 212,307 | 179,412 | 15% |
| nested_control.kern | 5,938 | 4,137 | 30% |
| polynomial.kern | 228 | 200 | 12% |
| relu_grid_stride.kern | 2,373 | 2,016 | 15% |
| saxpy_golden.kern | 112 | 86 | 23% |
| saxpy_random.kern | 4,280 | 3,274 | 24% |
| sum_rows.kern | 17,664 | 15,040 | 15% |
| sum_rows_golden.kern | 220 | 192 | 13% |
| **total** | **358,410** | **288,424** | **20%** |
