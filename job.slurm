#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=12000
#SBATCH --job-name="yolov7-training-with-augmentation"
# #SBATCH --output=test-trym.out
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:2
#SBATCH --constraint="V100"
# #SBATCH --gres=gpu:A100m40:1

AUGMENTATION_NUMBER=$1

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

echo "Augmentation number: ${AUGMENTATION_NUMBER}"

module purge
module load intel/2020b
module list

module load Python/3.8.6-GCCcore-10.2.0

pip install -r requirements.txt

mkdir job-environment-$SLURM_JOB_ID
cd job-environment-$SLURM_JOB_ID
git clone https://github.com/WongKinYiu/yolov7
cd yolov7
cp ../../augmentation.py .
cp ../../augmentation_transformations.py .
cp ../../torch_test.py .

python torch_test.py 1

pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

echo "running augmentation..."
python augmentation.py $AUGMENTATION_NUMBER
# python augmentation.py 1

echo "running training..."
# python train.py --batch-size 174 --epochs 1 --data Merged-sheep-dataset-3/data.yaml --weights 'yolov7-tiny.pt' --workers 2 --device 0,1 --img 640 640
python train.py --batch-size 80 --epochs 100 --data Merged-sheep-dataset-3/data.yaml --weights 'yolov7-tiny.pt' --workers 1 --device 0 --img 640 640

cp runs/ ../../runs/run-$SLURM_JOB_ID -r

cd ../..
# rm -rf job-environment-$SLURM_JOB_ID


echo "Job done"