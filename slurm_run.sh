#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH -t 08:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2005092
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

echo "START: $(date)"

rm logs/current.err
rm logs/current.out
ln -s $SLURM_JOBID.err logs/current.err
ln -s $SLURM_JOBID.out logs/current.out

source venv/bin/activate
export PYTHONPATH=venv/lib/python3.9/site-packages:$PYTHONPATH

srun python run.py 

echo "END: $(date)"
