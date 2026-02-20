#!/bin/bash
#SBATCH --job-name="lipopick-mpi"
#SBATCH --account=your_account
#SBATCH --partition=your_partition
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/project/logs/lipopick_%j.out
#SBATCH --error=/path/to/project/logs/lipopick_%j.err

# ============================================================
# lipopick MPI: multi-scale DoG particle picking
# Each MPI rank processes its share of micrographs independently.
# No --workers flag needed — 1 rank = 1 core.
# ============================================================

WORKDIR=/path/to/project
INPUT=${WORKDIR}/denoised_micrographs
OUTPUT=${WORKDIR}/outputs
THREADS=16

# Load modules (adjust for your cluster)
module load openmpi
module load miniconda3
source activate lipopick

# You may need to set LD_LIBRARY_PATH if mpi4py can't find libmpi.so.
# Find the path with: mpicc --showme:link | grep -oP '/\S+/lib'
# export LD_LIBRARY_PATH=/path/to/openmpi/lib:$LD_LIBRARY_PATH

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: ${THREADS}"
echo "Input: ${INPUT}"
echo ""

mkdir -p ${OUTPUT}
mkdir -p ${WORKDIR}/logs

echo "Running lipopick MPI picking..."
mpirun -np ${THREADS} lipopick-mpi \
    -i ${INPUT} \
    -o ${OUTPUT} \
    --dmin 50 --dmax 150 \
    --pixel-size 3.0341 \
    --threshold-percentile 80 \
    --nms-beta 1.3 \
    --refine

LIPOPICK_EXIT=$?

if [ $LIPOPICK_EXIT -eq 0 ]; then
    echo ""
    echo "lipopick completed successfully: $(date)"
    echo "Outputs: ${OUTPUT}"
else
    echo "lipopick FAILED with exit code $LIPOPICK_EXIT: $(date)"
    exit $LIPOPICK_EXIT
fi

echo "SBATCH JOB FINISHED"
