#!/usr/bin/bash
#SBATCH --job-name Eval
#SBATCH --account DD-24-8
#SBATCH --partition qgpu
#SBATCH --gpus 8
#SBATCH --time 00:30:00
#SBATCH --error /mnt/proj2/dd-24-8/mmdetection_work_dir/centerpoint_nus_eval/myJob.err
#SBATCH --output /mnt/proj2/dd-24-8/mmdetection_work_dir/centerpoint_nus_eval/myJob.out
#SBATCH --gres=gpu:8
#SBATCH --nodes 1

module load CUDA/11.7.0
module load Python/3.10.4-GCCcore-11.3.0
source $HOME/python_venv_2/bin/activate


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
	--nnodes=1 \
	--node_rank=0 \
	--master_addr="127.0.0.1" \
	--master_port=29500 \
	--nproc_per_node=8 \
    $HOME/3d_object_detection/src/mmdetection3d/tools/test.py \
    $HOME/3d_object_detection/src/mmdetection3d/configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
	/mnt/proj2/dd-24-8/torch_checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth \
    --launcher pytorch \
    --work-dir="/mnt/proj2/dd-24-8/mmdetection_work_dir/centerpoint_nus_eval"