
#!/bin/bash

module load python
module load profile/deeplrn

module load imagenet

export OMP_NUM_THREADS=8

cd $CINECA_SCRATCH/open_clip/src

source ../.env/bin/activate

torchrun --nproc_per_node 1 -m training.main \
	--train-data '/leonardo_scratch/large/userexternal/lbaraldi/Efficient_Foundation_Model_Training/CC3M/train/{00000..00331}.tar' \
	--train-num-samples 2905954 \
	--dataset-type webdataset \
	--batch-size 256 \
	--precision amp \
	--workers 8 \
	--model ViT-B-16_bcos \
	--imagenet-val ${IMAGENET2012_VAL/cineca/leonardo}/ \
	--report-to tensorboard
