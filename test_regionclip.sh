#!/bin/bash

#- Job parameters

# (TODO)
# Please modify job name

#SBATCH -J DomainAdptation            # The job name
#- SBATCH -o ret-%j.out        # Write the standard output to file named 'ret-<job_number>.out'
#- SBATCH -e ret-%j.err        # Write the standard error to file named 'ret-<job_number>.err'

#SBATCH -o ret_test.out        # Write the standard output to file named 'ret-<job_number>.out'
#SBATCH -e ret_test.err        # Write the standard error to file named 'ret-<job_number>.err'

#- Needed resources

# (TODO)
# Please modify your requirements

#SBATCH -p nv-gpu-hw             # Submit to 'nvidia-gpu' Partitiion or queue
# SBATCH -t 6-0:00:00             # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#- SBATCH -t 6-0:00:00             # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#SBATCH --nodes=1                 # Request N nodes
#SBATCH --gres=gpu:8              # Request M GPU per node
#- SBATCH --gres=gpu:4              # Request M GPU per node
#SBATCH --constraint=Volta        # Request GPU Type
#SBATCH --qos=gpu-normal

source /tools/module_env.sh
module list

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"

#export CUDA_VISIBLE_DEVICES=0,1,2,3,
echo "Use GPU ${CUDA_VISIBLE_DEVICES}$"

nvidia-smi 


source activate
conda activate detr
conda list pytorch
sh test_da_clip.sh


echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"

