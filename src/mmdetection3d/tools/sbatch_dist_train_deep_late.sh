#!/usr/bin/bash
#SBATCH --job-name LateFusion
#SBATCH --account DD-24-8
#SBATCH --partition qgpu
#SBATCH --gpus 8
#SBATCH --time 05:00:00
#SBATCH --error myJob_DeepLate_CustomSchedule_AVGPool.err
#SBATCH --output myJob_DeepLate_CustomSchedule_AVGPool.out
#SBATCH --gres=gpu:8
#SBATCH --nodes 1

module load CUDA/11.7.0
module load Python/3.10.4-GCCcore-11.3.0
source $HOME/python_venv_2/bin/activate

export TORCH_DISTRIBUTED_DEBUG="DETAIL"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
	--nnodes=1 \
	--node_rank=0 \
	--master_addr="127.0.0.1" \
	--master_port=29500 \
	--nproc_per_node=8 \
	$HOME/3d_object_detection/src/mmdetection3d/tools/train.py \
	$HOME/3d_object_detection/src/mmdetection3d/configs/deep_late_fusion/deep_late_fusion_custom_schedule.py \
	--launcher pytorch \
	--work-dir="/mnt/proj2/dd-24-8/mmdetection_work_dir/deep_late_fusion_cs_avg"
