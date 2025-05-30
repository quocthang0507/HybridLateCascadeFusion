#!/usr/bin/bash
#SBATCH --job-name Demo
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
python $HOME/3d_object_detection/src/late_fusion_clean_stereo_view_updated/demo.py \
	-output_dir $HOME/3d_object_detection/src/late_fusion_clean_stereo_view_updated/outputs \
	-img_path_left /mnt/proj2/dd-24-8/kitti_mmdet3d/training/image_2/000001.png \
	-img_path_right /mnt/proj2/dd-24-8/kitti_mmdet3d/training/image_3/000001.png \
    -lidar_path /mnt/proj2/dd-24-8/kitti_mmdet3d/training/velodyne/000001.bin \
	-calib_path /mnt/proj2/dd-24-8/kitti_mmdet3d/training/calib/000001.txt \
    -detector2d_config /home/it4i-carlos00/3d_object_detection/src/mmdetection/configs/base_faster_rcnn_fpn_3class_soft_nms.py \
	-detector2d_checkpoint_path /mnt/proj2/dd-24-8/mmdetection_work_dir/base_faster_rcnn_fpn_3class/best_coco_bbox_mAP_epoch_200.pth \
	-detector3d_checkpoint_path /mnt/proj2/dd-24-8/torch_checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth \
	-detector3d_config /home/it4i-carlos00/3d_object_detection/src/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class_low_nms.py \
	-frustum_detector3d_config /mnt/proj2/dd-24-8/mmdetection_work_dir/frustum_pointnet_dataset_v12/frustum_pointnet_one_hot_dynamic_sample_dataset_v12.py \
	-frustum_detector3d_checkpoint_path /mnt/proj2/dd-24-8/mmdetection_work_dir/frustum_pointnet_dataset_v12/best_IoU_3D_epoch_90.pth \
    -late_fusion_config /home/it4i-carlos00/3d_object_detection/src/late_fusion_clean_stereo_view_updated/late_fusion_cfg.json \
    -device cuda:0 \
    --visualize