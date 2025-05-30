#!/usr/bin/bash
#SBATCH --job-name FrustumDetector
#SBATCH --account DD-24-8
#SBATCH --partition qgpu
#SBATCH --gpus 8
#SBATCH --time 24:00:00
#SBATCH --error myJob_frustum_pointnet_nuscenes_mask_rcnn_6_rot.err
#SBATCH --output myJob_frustum_pointnet_nuscenes_mask_rcnn_6_rot.out
#SBATCH --gres=gpu:8
#SBATCH --nodes 1

module load CUDA/11.7.0
module load Python/3.10.4-GCCcore-11.3.0
source $HOME/python_venv_2/bin/activate

export TORCH_DISTRIBUTED_DEBUG="DETAIL"

export CUDA_LAUNCH_BLOCKING=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
# 	--nnodes=1 \
# 	--node_rank=0 \
# 	--master_addr="127.0.0.1" \
# 	--master_port=29500 \
# 	--nproc_per_node=8 \
# 	$HOME/3d_object_detection/src/mmdetection3d/tools/train.py \
# 	$HOME/3d_object_detection/src/mmdetection3d/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d-reduced-class.py \
# 	--launcher pytorch \
# 	--work-dir="/mnt/proj2/dd-24-8/mmdetection_work_dir/pointpillars_fpn_nuscenes_reduced"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
python -m torch.distributed.launch \
	--nnodes=1 \
	--node_rank=0 \
	--master_addr="127.0.0.1" \
	--master_port=29500 \
	--nproc_per_node=8 \
	$HOME/3d_object_detection/src/mmdetection3d/tools/train.py \
	$HOME/3d_object_detection/src/mmdetection3d/configs/frustum_localizer/frustum_pointnet_nuscenes_mask_rcnn_6_rot.py \
	--launcher pytorch \
	--work-dir="/mnt/proj2/dd-24-8/mmdetection_work_dir/frustum_pointnet_nuscenes_mask_rcnn_6_rot"