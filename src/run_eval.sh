export CUDA_VISIBLE_DEVICES=1
python $HOME/HybridLateCascadeFusion/src/eval.py \
	-output_dir $HOME/experiments/code_to_publish \
	-kitti_root_path /DATA/kitti_mmdet3d/training \
	-annotation_file_eval /DATA/kitti_mmdet3d/kitti_infos_val.pkl \
	-validation_split_path /DATA/kitti_mmdet3d/ImageSets/val.txt \
	-late_fusion_config /home/carlos00/HybridLateCascadeFusion/src/example_cfg.json \
    -device cuda:0
