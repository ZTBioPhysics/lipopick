#!/bin/bash
#SBATCH --job-name=lipopick-mpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:30:00
#SBATCH --output=lipopick-mpi_%j.out
#SBATCH --error=lipopick-mpi_%j.err

# ── Load modules (adjust for your HPC) ──────────────────────────────
# module load openmpi/4.1
# module load anaconda3
# conda activate lipopick

# ── Run lipopick with MPI ────────────────────────────────────────────
# Each rank processes its share of micrographs independently.
# No --workers flag needed — 1 rank = 1 core.

mpirun -np 32 lipopick-mpi \
    -i /path/to/micrographs/ \
    -o /path/to/outputs/ \
    --dmin 50 --dmax 150 \
    --pixel-size 3.0341 \
    --threshold-percentile 80 \
    --nms-beta 1.3 \
    --refine
