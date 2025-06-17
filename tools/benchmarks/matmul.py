#!/usr/bin/env python3

import subprocess
import itertools
import argparse
import sys

# Matrix dimensions
MATRIX_SIZES = [
    (4096, 4096),
    (11008, 4096),
    (4096, 11008),
    (8192, 8192),
    (8192, 28672),
    (28672, 8192)
]

# Batch sizes
BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Default benchmark parameters
WARMUP = 5
REPEAT = 20
BATCH = 1
DEFAULT_BACKEND = "petit"
ALGO = "tune"
DEFAULT_ATYPE = "fp16"
DEFAULT_CTYPE = "fp16"

def run_benchmark(m, n, k, args):
    cmd = [
        "bench_matmul",
        "-backend", f"{args.backend}",
        "-atype", f"{args.atype}",
        "-ctype", f"{args.ctype}",
        "-m", f"{m}",
        "-k", f"{k}",
        "-n", f"{n}",
        "-warmup", f"{WARMUP}",
        "-repeat", f"{REPEAT}",
        "-batch", f"{BATCH}",
        "-algo", f"{ALGO}"
    ]
    
    cmd_str = " ".join(cmd)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to run benchmark: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description='Run matrix multiplication benchmarks')
    parser.add_argument('--backend', type=str, default=DEFAULT_BACKEND,
                      help=f'Backend to use for matrix multiplication (default: {DEFAULT_BACKEND})')
    parser.add_argument('--atype', type=str, default=DEFAULT_ATYPE,
                      help=f'Data type of matrix A (default: {DEFAULT_ATYPE})')
    parser.add_argument('--ctype', type=str, default=DEFAULT_CTYPE,
                      help=f'Data type of matrix C (default: {DEFAULT_CTYPE})')
    
    args = parser.parse_args()
    
    for n, k in MATRIX_SIZES:
        for m in BATCH_SIZES:
            run_benchmark(m, n, k, args)

if __name__ == "__main__":
    main()
