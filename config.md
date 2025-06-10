# Configuration

To run the code, you should provide a configuration file in a .json format. You can find an example [here](./src/example_cfg.json). The configuration file should have the following structure:

```
{
    "score_thr_2d": [0.5, 0.4, 0.5],
    "score_thr_3d": [0.2, 0.3, 0.3],
    "bbox_matching_iou_thr": 0.4,
    "bbox_matching_mode": "iou",
    "detection_recovery_iou_thr": 0.3,
    "min_pts_frustum": 10,

    "align_frustum": true,
    "use_gaussian_likelihoods": true,
    "enlarge_factor": 0.05,

    "use_detection_recovery": true,
    "use_label_fusion": true,
    "use_score_fusion": true,
    "use_final_nms": false,
    "keep_oov_bboxes": false,
    "class_prior": [0.33, 0.33, 0.33],

    "num_classes": 3,
    "classes": ["Pedestrian", "Cyclis", "Car"],

    "valid_2d_classes": [0, 1, 2],
    "img_label_mapping": [2, 0, 1],

    "final_nms_cfg": {
        "thresh": 0.3,
        "pre_max_size": 100,
        "post_max_size": 100
    },

    "visualization_cfg": {
        "colors": [
            [178, 255, 102],
            [255, 102, 178],
            [153, 255, 255]
        ],
        "fill_bboxes_2d": true,
        "alpha": 50
    },

    "detector2d": {
        "cfg_path": "/path/to/rgb_detector/cfg.py",
        "checkpoint_path": "path/to/rgb_detector/checkpoint.pth"
    },

    "detector3d": {
        "cfg_path": "/path/to/lidar_detector/cfg.py",
        "checkpoint_path": "path/to/lidar_detector/checkpoint.pth"
    },

    "frustum_detector": {
        "cfg_path": "/path/to/frustum_localizer/cfg.py",
        "checkpoint_path": "path/to/frustum_localizer/checkpoint.pth"
    }
}
```

Here a description of the most important parameters
- ```score_thr_2d```: the score threshold for 2D detector of the RGB branch, for each class (already mapped to LiDAR indices)
- ```score_thr_3d```: the score threshold for 3D detector of the LiDAR branch, for each class
- ```bbox_matching_iou_thr```: the IoU threshold for the Bounding Box Matching, matches with a IoU lower than this will be removed
- ```bbox_matching_mode```: the way the IoU is computed, can be iou, giou or iof (see [mmdet](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/structures/bbox/bbox_overlaps.py))
- ```detection_recovery_iou_thr```: the IoU threshold for the Detection Recovery, 3D bounding boxes whose projection do not have a sufficient IoU with the 2D ones that were used to build the frustum will not be confirmed
- ```min_pts_frustum```: the minimum number of points inside a Frustum Proposal to be processed
- ```align_frustum```: true if the Frustum Localizer was trained on normalized coordinates (true for the provided model)
- ```use_gaussian_likelihoods```: true if the gaussian likelihood was added during training of the Frustum Localizer (true for the provided model)
- ```valid_2d_classes```: the indices of the valid classes predicted by the 2D network. Use this if the 2D model predict a bigger set of classes
- ```img_label_mapping```: a mapping to match 2D classes with 3D classes, if element i of this list have a value x, it means that class i of the RGB detector is equivalent to class x of the LiDAR detector. For example, the Faster RCNN that we used was trained with a different order of the classes, i.e. (Car, Pedestrian, Cyclists) instead of (Pedestrian, Cyclist, Car) which is the usual for mmdetection3d models.
