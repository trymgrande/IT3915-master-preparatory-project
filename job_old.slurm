#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=12000
#SBATCH --job-name="baseline-test"
# #SBATCH --output=test-trym.out
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:2
#SBATCH --constraint="V100"
# #SBATCH --gres=gpu:A100m40:1



WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load intel/2020b
module list


cd yolov7


module load Python/3.8.6-GCCcore-10.2.0

# python -m pip install -r requirements.txt
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# python3 -m wandb.login()

python torch_test.py

# python train.py --batch-size 174 --epochs 100 --data Merged-sheep-dataset-3/data.yaml --weights 'yolov7-tiny.pt' --workers 2 --device 0,1 --img 640 640
python train.py --batch-size 86 --epochs 1 --data Merged-sheep-dataset-3/data.yaml --weights 'yolov7-tiny.pt' --workers 1 --device 0 --img 640 640