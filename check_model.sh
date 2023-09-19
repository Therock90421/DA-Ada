#!/bin/bash

#- Job parameters

# (TODO)
# Please modify job name

#SBATCH -J DomainAdptation            # The job name
#- SBATCH -o ret-%j.out        # Write the standard output to file named 'ret-<job_number>.out'
#- SBATCH -e ret-%j.err        # Write the standard error to file named 'ret-<job_number>.err'

#SBATCH -o check_test.out        # Write the standard output to file named 'ret-<job_number>.out'
#SBATCH -e check_test.err        # Write the standard error to file named 'ret-<job_number>.err'

#- Needed resources

# (TODO)
# Please modify your requirements

#SBATCH -p nv-gpu             # Submit to 'nvidia-gpu' Partitiion or queue
# SBATCH -t 6-0:00:00             # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#- SBATCH -t 6-0:00:00             # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#SBATCH --nodes=1                 # Request N nodes
#SBATCH --gres=gpu:1              # Request M GPU per node
#- SBATCH --gres=gpu:4              # Request M GPU per node
#SBATCH --constraint=Volta        # Request GPU Type
#SBATCH --qos=gpu-long

source /tools/module_env.sh
module list

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"

module load cuda-cudnn/11.0-8.0.5


echo "Use GPU ${CUDA_VISIBLE_DEVICES}$"

nvidia-smi 


source activate
conda activate detr
echo $(which gcc)
#cd SRFBN
#- python train.py -opt options/train/train_SRFBN.json
#- python test.py -opt options/test/test_SRFBN_x4_BI.json
#python test.py -opt options/test/test_SRFBN_example.json
conda list pytorch
python check_equal.py


echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"

