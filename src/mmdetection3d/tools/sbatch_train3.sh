#!/usr/bin/bash
#SBATCH --job-name TrainFrustumDetector
#SBATCH --account DD-24-8
#SBATCH --partition qgpu
#SBATCH --gpus 1
#SBATCH --time 48:00:00
#SBATCH --error myJob_frustum_pointnet_nuscenes_mask_rcnn_low_points_4_rot_s.err
#SBATCH --output myJob_frustum_pointnet_nuscenes_mask_rcnn_low_points_4_rot_s.out
#SBATCH --gres=gpu:1
#SBATCH --nodes 1

module load CUDA/11.7.0
module load Python/3.10.4-GCCcore-11.3.0
source $HOME/python_venv_2/bin/activate

export TORCH_DISTRIBUTED_DEBUG="DETAIL"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python $HOME/3d_object_detection/src/mmdetection3d/tools/train.py \
	$HOME/3d_object_detection/src/mmdetection3d/configs/frustum_localizer/frustum_pointnet_nuscenes_mask_rcnn_low_points_4_rot.py \
	--work-dir="/mnt/proj2/dd-24-8/mmdetection_work_dir/frustum_pointnet_nuscenes_mask_rcnn_low_points_4_rot_single_gpu"
