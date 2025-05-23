#!/bin/bash
#$ -l h_rt=240:0:0
#$ -l h_vmem=20G
#$ -pe smp 48
#$ -l gpu=4
#$ -l gpu_type=ampere
#$ -l gpuhighmem
#$ -cwd
#$ -j y
#$ -N amos_mm_stage1
#$ -l rocky
#$ -l node_type=rdg
set -e

bash
module load miniforge
module load cuda/11.8.0-gcc-12.2.0

source activate med3dllm
cd /data/home/qp24681/u2Tokenizer
sh /data/home/qp24681/u2Tokenizer/hpc/amos_mm_stage1/amos_mm_diffts_stage1.sh
sh /data/home/qp24681/u2Tokenizer/hpc/amos_mm_stage1/amos_mm_dmtp_stage1.sh
sh /data/home/qp24681/u2Tokenizer/hpc/amos_mm_stage1/amos_mm_mu2_stage1.sh
sh /data/home/qp24681/u2Tokenizer/hpc/amos_mm_stage1/amos_mm_linvt_stage1.sh
sh /data/home/qp24681/u2Tokenizer/hpc/amos_mm_stage1/amos_mm_rpe_stage1.sh