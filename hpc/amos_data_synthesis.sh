#!/bin/bash
#$ -l h_rt=240:0:0
#$ -l h_vmem=20G
#$ -pe smp 24
#$ -l gpu=2
#$ -l gpuhighmem
#$ -cwd
#$ -j y
#$ -N amos_thinking_data_synthesis
#$ -l rocky
#$ -m beas
#$ -o log/

set -e

bash
module load miniforge
module load cuda/11.8.0-gcc-12.2.0

source activate vllm
cd /data/home/qp24681/u2Tokenizer
echo $SGE_HGR_gpu 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.preprocess.aatlas_amos_thinking_data_synthesis