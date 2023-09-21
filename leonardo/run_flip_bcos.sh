#!/bin/bash -x
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH -t 24:00:00
#SBATCH -o job_%A.out
#SBATCH -e job_%A.err
#SBATCH --wait-all-nodes=1
#SBATCH -p boost_usr_prod
#SBATCH -A tra23_ELLIS 
#SBATCH --reservation s_tra_Ellis

module load python
module load profile/deeplrn

module load imagenet

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd $CINECA_SCRATCH/open_clip/src
source ../.env/bin/activate

srun --cpu_bind=v --accel-bind=gn python -u training/main.py \
	--report-to tensorboard \
	--save-frequency 1 \
	--train-data '/leonardo_scratch/large/userexternal/lbaraldi/Efficient_Foundation_Model_Training/CC3M/train/{00000..00331}.tar' \
	--train-num-samples 2905954 \
	--dataset-type webdataset \
	--batch-size 256 \
	--precision amp \
	--workers 8 \
	--model $1 \
	--warmup 2000 \
	--imagenet-val ${IMAGENET2012_VAL/cineca/leonardo}/ \
	--local-loss \
	--gather-with-grad \
	--zeroshot-frequency 1
