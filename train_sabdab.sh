#!/bin/zsh

########## adjust configs according to your needs ##########
DATA_DIR=./data/sabdab
TRAIN_SET=${DATA_DIR}/train.txt
VALID_SET=${DATA_DIR}/val.txt
SAVE_DIR=${DATA_DIR}/ckpt

BATCH_SIZE=4
DATASET=SabDab
######### end of adjust ##########

########## Instruction ##########
# This script takes three optional environment variables:
# MODEL / GPU / ADDR / PORT
# e.g. Use gpu 0, 1 and 4 for training, set distributed training
# master address and port to localhost:9901, the command is as follows:
#
# MODEL=MEAN GPU="0,1,4" ADDR=localhost PORT=9901 bash train.sh
#
# Default value: GPU=-1 (use cpu only), ADDR=localhost, PORT=9901
# Note that if your want to run multiple distributed training tasks,
# either the addresses or ports should be different between
# each pair of tasks.
######### end of instruction ##########

# set master address and port e.g. ADDR=localhost PORT=9901 bash train.sh
MASTER_ADDR=localhost
MASTER_PORT=9901
if [ $ADDR ]; then MASTER_ADDR=$ADDR; fi
if [ $PORT ]; then MASTER_PORT=$PORT; fi
echo "Master address: ${MASTER_ADDR}, Master port: ${MASTER_PORT}"

# set model
if [ -z "$MODEL" ]; then
	MODEL=ElliDock
fi

# set gpu, e.g. GPU="0,1,2,3" bash train.sh
if [ -z "$GPU" ]; then
    GPU="-1"  # use CPU
fi
export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPUs: $GPU"
GPU_ARR=(`echo $GPU | tr ',' ' '`)

if [ ${#GPU_ARR[@]} -gt 1 ]; then
    export OMP_NUM_THREADS=2
	  PREFIX="python3 -m torch.distributed.run --nproc_per_node=${#GPU_ARR[@]} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --standalone --nnodes=1"
else
    PREFIX="python"
fi

${PREFIX} train.py \
    --dataset $DATASET \
    --train_set $TRAIN_SET \
    --valid_set $VALID_SET \
    --save_dir $SAVE_DIR \
    --model_type $MODEL \
    --lr 2e-4 \
    --max_epoch 500 \
    --patience 8 \
    --save_topk 10 \
    --embed_dim 64 \
    --hidden_size 128 \
    --k_neighbors 10 \
    --n_layers 2 \
    --att_heads 16 \
    --rbf_dim 20 \
    --shuffle \
    --batch_size ${BATCH_SIZE} \
    --gpu "${!GPU_ARR[@]}"
