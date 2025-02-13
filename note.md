# CUDA Execution Model Summary

## Threads
- Smallest unit of execution.
- Execute the same code on different data.
- Organized into blocks.

## Blocks
- Group of threads sharing data and synchronizing execution.
- Unique ID within the grid.
- Optimal thread count per block: multiple of 32 (e.g., 128, 256, 512).

## Grids
- Collection of blocks executing a kernel function.
- Unique ID.
- Number of blocks = ceil(total elements / threads per block).

## Summary
1. **Threads per Block**
    - Organized inside blocks.
    - Multiple of 32 (warp size).
    - Common choices: 128, 256, 512.

2. **Blocks per Grid**
    - Number of blocks = ceil(total elements / threads per block).

3. **Shared Memory**
    - Ensure total shared memory per block is within GPU limits.
