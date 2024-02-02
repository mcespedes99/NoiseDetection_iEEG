#!/bin/bash

module load python
source /scratch/mcesped/code/venv_ML/bin/activate
pip install --no-index torch torchvision

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "starting training..."
python /scratch/mcesped/code/NoiseDetection_iEEG/interictal_classifier/train.py

