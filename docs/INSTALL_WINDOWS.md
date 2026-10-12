# Installing and running KernC on Windows — step by step

This guide assumes a fresh Windows 10/11 (64-bit) machine. Parts 1–4 work on
**any** PC, with or without an NVIDIA GPU. Part 5 runs the generated PTX on a
real NVIDIA GPU. Part 6 is the no-GPU alternative (Google Colab).

Time needed: about 30–45 minutes, mostly downloads.

---

## Part 1 — Install the tools (one time)

### Step 1. Install Python 3.10 or newer

1. Go to <https://www.python.org/downloads/windows/> and download the latest
   **Windows installer (64-bit)**.
2. Run it. On the first screen, **tick "Add python.exe to PATH"**, then click
   **Install Now**.
3. Open a new **Command Prompt** (Start → type `cmd`) and check:
   ```
   python --version
   ```
   You should see `Python 3.1x.x`. If Windows opens the Microsoft Store instead,
   use `py --version`, and type `py` wherever this guide says `python`.

### Step 2. Install the C++ compiler and CMake (Visual Studio Build Tools)

1. Go to <https://visualstudio.microsoft.com/downloads/>. Download
   **Visual Studio 2022 Community** (or scroll down to *Tools for Visual Studio*
   → **Build Tools for Visual Studio 2022** — smaller, no IDE).
2. Run the installer. In the **Workloads** tab, tick
   **"Desktop development with C++"**. On the right, make sure these stay ticked:
   * MSVC v143 – VS 2022 C++ x64/x86 build tools
   * Windows 11 (or 10) SDK
   * **C++ CMake tools for Windows**
3. Click **Install** (≈ 5–8 GB) and wait. Restart if asked.

### Step 3. Open the right terminal

From now on, always use **Developer PowerShell for VS 2022**
(Start menu → type "Developer PowerShell"). It has the compiler and CMake on
its PATH; a normal PowerShell does not. Check:
```
cmake --version
cl
```
`cmake` prints a version (3.2x or newer); `cl` prints "Microsoft (R) C/C++
Optimizing Compiler". If `cmake` is not recognized, re-run the Visual Studio
Installer → **Modify** → make sure **C++ CMake tools for Windows** is ticked.

---

## Part 2 — Build KernC

### Step 4. Extract the project

1. Right-click `kernc.zip` → **Extract All…** → choose `C:\Projects`.
2. You should now have `C:\Projects\kernc\` containing `CMakeLists.txt`,
   `src\`, `tests\`, `tools\`, `README.md`.
   (If you got `C:\Projects\kernc\kernc\`, move the inner folder up one level.)

> Tip: avoid paths with spaces or OneDrive-synced folders for build trees.

### Step 5. Configure and build

In **Developer PowerShell for VS 2022**:
```
cd C:\Projects\kernc
cmake -S . -B build
cmake --build build --config Release
```
The first command detects Visual Studio; the second compiles everything.
It ends with lines like `kernc.vcxproj -> C:\Projects\kernc\build\bin\kernc.exe`.
Warnings (yellow text) are fine; **errors** are not — see Troubleshooting.

> Shortcut: steps 5–8 are also done by double-clicking **`build.bat`**
> (run it from the Developer PowerShell so it can find CMake: `.\build.bat`).

### Step 6. Check the compiler runs

```
build\bin\kernc.exe --version
build\bin\kernc.exe --help
```
Expected: `kernc 1.0.0`, then the usage text.

### Step 7. Run the C++ unit tests

```
build\bin\kernc_unit_tests.exe
```
Expected last line: `24 tests, 84 checks, 0 failures`.

---

## Part 3 — Use the compiler

### Step 8. Compile your first kernel to PTX

```
build\bin\kernc.exe compile examples\saxpy.kern -O1 -o saxpy.ptx
notepad saxpy.ptx
```
You will see NVIDIA PTX assembly: `.visible .entry saxpy(`, `mad.lo.s32`,
`ld.global.f32`, `mul.rn.f32`, …

Look inside the compiler:
```
build\bin\kernc.exe compile examples\saxpy.kern --dump-tokens
build\bin\kernc.exe compile examples\saxpy.kern --dump-ast
build\bin\kernc.exe compile examples\saxpy.kern -O0 --dump-ir
build\bin\kernc.exe compile examples\saxpy.kern -O1 --dump-ir --stats
```
Compare the `-O0` and `-O1` IR: constants folded, copies removed, the address
of `y[i]` computed once, and (in loops) invariant code hoisted.

See an error message:
```
build\bin\kernc.exe check tests\errors\mixed_arith.kern
```
→ `tests\errors\mixed_arith.kern:4:16: error: cannot apply '+' to f32 and i32`

### Step 9. Run a kernel on the CPU reference interpreters

```
python tools\make_launch.py tests\exec\saxpy_golden.kern > saxpy.launch
build\bin\kernc.exe run tests\exec\saxpy_golden.kern --launch saxpy.launch --engine ast
build\bin\kernc.exe run tests\exec\saxpy_golden.kern --launch saxpy.launch --engine ir -O1
```
Both print `status ok` and the output buffers as hex words; they must be identical.

### Step 10. The demo

```
python tools\demo.py
```
Draws the Mandelbrot set computed by the compiled kernel and confirms that the
AST oracle and the optimised IR agree on every pixel.

---

## Part 4 — Run the verification suite (no GPU needed)

### Step 11. (Recommended) get `ptxas` to validate the generated PTX

`ptxas` is NVIDIA's PTX assembler. It runs **without a GPU**. Easiest way:
```
python -m pip install nvidia-cuda-nvcc-cu12
```
(Alternatively install the full CUDA Toolkit, Part 5 step 15.) The test tools
find `ptxas` automatically in `PATH`, in `%CUDA_PATH%\bin`, or in that pip
package; or set it explicitly:
`$env:PTXAS = "C:\path\to\ptxas.exe"`. Without ptxas the suite still runs and
just skips PTX validation.

### Step 12. Run all tests

```
python tools\run_tests.py
```
It prints which `kernc` and `ptxas` it uses, then
`86/86 passed - all good` (unit + errors + codegen + exec + edge).
Use `-v` to list every test, `--only exec` to run one layer.

### Step 13. Fuzz the compiler

```
python tools\fuzz.py -n 300
```
Generates 300 random, UB-free programs and checks that all engines agree and
that ptxas accepts every PTX file. Expected: `300 programs in …s: 0 failures`.
Any failure is saved to `fuzz_failures\` and automatically reduced.

### Step 14. Mutation testing and benchmarks

```
python tools\mutate.py
python tools\bench.py
```
`mutate.py` injects 16 known bugs into the compiler one at a time and reports
how many the suite catches (expected: `16/16 = 100%`), writing
`MUTATION_REPORT.md`. `bench.py` prints instruction counts at `-O0` vs `-O1`.

---

## Part 5 — Run on a real NVIDIA GPU (optional)

Needs an NVIDIA GPU of the GTX 10-series or newer.

### Step 15. Driver, CUDA Toolkit and CuPy

1. Install the latest **NVIDIA driver** for your GPU:
   <https://www.nvidia.com/Download/index.aspx>. Then check in any terminal:
   ```
   nvidia-smi
   ```
   The top-right shows `CUDA Version: 12.x` (the newest CUDA the driver supports).
2. (Optional if you did step 11) install the **CUDA Toolkit 12.x**:
   <https://developer.nvidia.com/cuda-downloads> → Windows → x86_64 → exe (local).
   A *Custom* install with only "CUDA → Development → Compiler" is enough.
3. Install CuPy matching CUDA 12:
   ```
   python -m pip install cupy-cuda12x numpy
   python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), cupy.cuda.Device().compute_capability)"
   ```
   Expected: `1 86` (or your GPU's compute capability).
   If `nvidia-smi` showed `CUDA Version: 11.x`, use `cupy-cuda11x` instead.

### Step 16. GPU tests

```
python tools\gpu_run.py tests\exec\saxpy_golden.kern
python tools\run_tests.py --gpu
python tools\fuzz.py -n 200 --gpu
python tools\demo.py
```
Every exec/edge test and every fuzz program now also runs on the GPU at `-O0`
and `-O1`, and the GPU result must match the AST oracle bit-for-bit.
`demo.py` additionally prints `AST oracle vs GPU (-O1 PTX): identical`.

KernC targets `sm_75` by default; `gpu_run.py` automatically picks a lower
target on older GPUs. For your own PTX use `--arch sm_86` (RTX 30xx),
`--arch sm_89` (RTX 40xx), etc.

---

## Part 6 — No NVIDIA GPU? Use Google Colab (free)

1. Open <https://colab.research.google.com> → **File → Upload notebook** →
   choose `notebooks\KernC_Colab.ipynb` from the project.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run the cells top to bottom. The upload cell asks for `kernc.zip`; pick the
   same zip you extracted. The notebook builds KernC, runs the full suite with
   `--gpu`, fuzzes on the GPU and runs mutation testing.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `cmake` / `cl` is not recognized | You are in a normal terminal. Use **Developer PowerShell for VS 2022**. |
| `No CMAKE_CXX_COMPILER could be found` | The "Desktop development with C++" workload is missing — Visual Studio Installer → Modify. |
| `python` opens the Microsoft Store | Use `py` instead, or reinstall Python with "Add python.exe to PATH". |
| `error: kernc executable not found` | Build first (step 5). Run the tools from the project folder `C:\Projects\kernc`. Or set `$env:KERNC_BIN = "C:\...\kernc.exe"`. |
| `ptxas: not found (PTX validation skipped)` | Step 11, or set `$env:PTXAS` to the full path of `ptxas.exe`. |
| CuPy: `DLL load failed` / `CUDA driver version is insufficient` | The CuPy package must match your driver: `cupy-cuda12x` needs `CUDA Version: 12.x` in `nvidia-smi`. Update the driver. |
| `--gpu given but CuPy cannot see a CUDA GPU` | No NVIDIA GPU, or the driver is missing. Use Colab (Part 6). |
| Build is very slow / files locked | Antivirus scanning `build\`; add an exclusion for the project folder. |
| Stale build after editing sources | `cmake --build build --config Release` again; for a clean rebuild delete the `build` folder and repeat step 5. |
