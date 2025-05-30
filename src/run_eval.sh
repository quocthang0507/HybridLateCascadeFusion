#!/usr/bin/bash
#SBATCH --job-name ExpertLateFusion
#SBATCH --account DD-24-8
#SBATCH --partition qgpu
#SBATCH --gpus 1
#SBATCH --time 2:00:00
#SBATCH --error myJob_pp.err
#SBATCH --output myJob_pp.out
#SBATCH --gres=gpu:1
#SBATCH --nodes 1

module load CUDA/11.7.0
module load Python/3.10.4-GCCcore-11.3.0
source $HOME/python_venv_2/bin/activate

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
export CUDA_VISIBLE_DEVICES=0
python $HOME/HybridLateCascadeFusion/src/eval.py \
	-output_dir $HOME/experiments/comparison/code_to_publish \
	-kitti_root_path /mnt/proj2/dd-24-8/kitti_mmdet3d/training \
	-faster_config_path /home/it4i-carlos00/3d_object_detection/src/mmdetection/configs/base_faster_rcnn_fpn_3class_soft_nms.py \
	-faster_checkpoint_path /mnt/proj2/dd-24-8/mmdetection_work_dir/base_faster_rcnn_fpn_3class/best_coco_bbox_mAP_epoch_200.pth \
	-detector3d_checkpoint_path /mnt/proj2/dd-24-8/torch_checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth \
	-detector3d_config /home/it4i-carlos00/3d_object_detection/src/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class_low_nms.py \
	-frustum_detector3d_config /mnt/proj2/dd-24-8/mmdetection_work_dir/frustum_pointnet_single_view/frustum_pointnet_single_one_hot_dynamic_sample_dataset.py \
	-frustum_detector3d_checkpoint_path /mnt/proj2/dd-24-8/mmdetection_work_dir/frustum_pointnet_single_view/best_IoU_3D_epoch_76.pth \
	-annotation_file_eval /mnt/proj2/dd-24-8/kitti_mmdet3d/kitti_infos_val.pkl \
	-validation_split_path /mnt/proj2/dd-24-8/kitti_mmdet3d/ImageSets/val.txt \
	-late_fusion_config /home/it4i-carlos00/HybridLateCascadeFusion/src/late_fusion_cfg.json \
    -device cuda:0
