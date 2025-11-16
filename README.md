# benchpress - Compiler Benchmark Harness Generator

## Overview

`benchpress` is a tool that generates self-building benchmark harnesses for C code. It takes a template file with a few annotated functions and produces a single C file that compiles the benchmark function with multiple compiler configurations and measures their performance.

## Use Cases

- Compare how GCC vs Clang optimize specific code
- Test different optimization levels (-O2 vs -O3)
- Evaluate the impact of specific compiler flags
- Create reproducible benchmarks for bug reports
- Share single-file benchmarks that build and run themselves

## Quick Start

### 1. Write a template with three markers

```c
// matmul_template.c
#include <stdint.h>
#include <time.h>

typedef struct { float data[16]; } Matrix4x4;

void init_matrix(Matrix4x4 *m) {
    for (int i = 0; i < 16; i++) {
        m->data[i] = (float)(i + 1);
    }
}

// BENCHFUNC: Function to compile with different compilers/flags
BENCHFUNC void matmul(Matrix4x4 *a, Matrix4x4 *b, Matrix4x4 *result) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += a->data[i*4+k] * b->data[k*4+j];
            }
            result->data[i*4+j] = sum;
        }
    }
}

// WARMUP: Called once before benchmarking each config
// Must be named {benchmark_name}_warmup to pair with BENCHMARK
WARMUP void run_benchmark_warmup(void) {
    Matrix4x4 a, b, result;
    init_matrix(&a);
    init_matrix(&b);
    for (int64_t i = 0; i < 1000000; i++) {
        matmul(&a, &b, &result);
    }
}

// BENCHMARK: Runs benchmark iterations
BENCHMARK void run_benchmark(void) {
    Matrix4x4 a, b, result;
    init_matrix(&a);
    init_matrix(&b);
    for (int64_t i = 0; i < 1000000000; i++) {
        matmul(&a, &b, &result);
    }
}
```

- `BENCHFUNC` - Function(s) to compile with different compilers/flags (can have multiple)
- `WARMUP` - Warmup code run before each benchmark (must be named `{benchmark_name}_warmup`)
- `BENCHMARK` - Benchmark iterations (harness handles timing, can have multiple)

### 2. Generate the benchmark

```bash
python3 benchpress.py matmul_template.c \
  --compilers gcc:clang \
  --flags="-O2:-O3" \
  -o benchmark.c
```

### 3. Run it

```bash
sh benchmark.c
```

Output:
```
GCC version: gcc (GCC) 15.2.0
Clang version: clang version 20.1.6

gcc -O2: 6.374 seconds
gcc -O3: 3.681 seconds
clang -O2: 6.163 seconds
clang -O3: 3.801 seconds
gcc -O2 vs clang -O2: clang -O2 was 1.03x faster
clang -O3 vs gcc -O3: gcc -O3 was 1.03x faster
```

The generated `benchmark.c` is self-contained and can be shared as a single file.

## Usage

### Basic Usage

```bash
# Compare gcc -O3 vs clang -O3 (default)
benchpress template.c -o benchmark.c

# Run the generated benchmark
sh benchmark.c
```

### Flag Combinations

Generate all combinations of compilers and flag sets:

```bash
benchpress.py template.c \
  --compilers gcc:clang \
  --flags="-O2:-O3:-O3 -march=native" \
  -o benchmark.c
```

This creates 6 configurations:
- gcc -O2, gcc -O3, gcc -O3 -march=native
- clang -O2, clang -O3, clang -O3 -march=native

### Individual Configs

Specify each configuration manually:

```bash
benchpress.py template.c \
  --config gcc:-O2 \
  --config "gcc:-O3 -march=native" \
  --config clang:-O2 \
  --config clang:-O3 \
  -o benchmark.c
```

Configuration format: `COMPILER:FLAGS`
- `COMPILER` must be `gcc` or `clang`
- `FLAGS` are compiler flags (quote if they contain spaces)

### Custom Comparisons

By default, flag combinations compare same flags across different compilers. You can override this with `--compare`:

```bash
# Compare specific configs (disables default comparisons)
benchpress.py template.c \
  --compilers gcc:clang \
  --flags="-O2:-O3" \
  --compare "gcc -O3,clang -O3" \
  --compare "gcc -O2,clang -O2" \
  -o benchmark.c

# Compare optimization levels
benchpress.py template.c \
  --config gcc:-O2 \
  --config gcc:-O3 \
  --compare "gcc -O2,gcc -O3" \
  -o benchmark.c

# Test impact of -march=native
benchpress.py template.c \
  --config "gcc:-O3" \
  --config "gcc:-O3 -march=native" \
  --compare "gcc -O3,gcc -O3 -march=native" \
  -o benchmark.c
```

The `--compare` flag can be repeated to create multiple comparison groups. When used, it disables the default cross-compiler comparisons.

## Command-Line Reference

```
usage: benchpress.py [-h] -o OUTPUT [--config SPEC] [--compilers LIST]
                     [--flags LIST] [--compare LABELS]
                     input

positional arguments:
  input                 template file with BENCHFUNC/WARMUP/BENCHMARK markers

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output self-building benchmark file
  --config SPEC         add config: compiler:flags (can be repeated)
  --compilers LIST      compilers to test, colon-separated (e.g., gcc:clang)
  --flags LIST          flag sets to test, colon-separated (e.g., "-O2:-O3")
  --compare LABELS      specific configs to compare, comma-separated
```

## Requirements

- Python 3+
- `pycparser`
- `fake_libc_include` directory (from pycparser GitHub repo)
- GCC and/or Clang installed on the system (to run the outputs)

## Examples

### Compare optimization levels
```bash
benchpress.py mycode.c --compilers gcc:clang --flags="-O0:-O1:-O2:-O3" -o bench.c
```

### Test architecture-specific flags
```bash
benchpress.py mycode.c --compilers gcc \
  --flags="-O3:-O3 -march=native:-O3 -march=skylake" -o bench.c
```

### Share reproducible benchmarks
```bash
# Generate benchmark
benchpress.py issue_report.c --compilers gcc:clang --flags="-O2:-O3" -o repro.c

# Attach repro.c to bug report
# Anyone can run it with: sh repro.c
```

## Limitations

- C++ doesn't work
- Requires a posix shell