#!/bin/bash
#SBATCH --job-name="test_4worker"
#SBATCH --workdir=.
#SBATCH --output=serial_%j.out
#SBATCH --error=serial_%j.err
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
python3 eddl_train_batch_compss_cifar10_sync.py --num_workers 4 
--num_epochs 1 --num_epochs_for_param_sync 1 --workers_batch_size 250 > serial.out
