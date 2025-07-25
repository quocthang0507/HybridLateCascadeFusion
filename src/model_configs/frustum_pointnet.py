auto_scale_lr = dict(base_batch_size=48, enable=False)
backend_args = None
class_names = [
    'Pedestrian',
    'Cyclist',
    'Car',
]
data_root = '/path/to/frustum_datasets/'
dataset_type = 'FrustumDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1, rule='greater', save_best='IoU_3D', type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
epoch_num = 100
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromDict',
        use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        flip=False,
        img_scale=(
            1333,
            800,
        ),
        pts_scale_ratio=1,
        transforms=[
            dict(
                rot_range=[
                    0,
                    0,
                ],
                scale_ratio_range=[
                    1.0,
                    1.0,
                ],
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTrans'),
            dict(type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    -100,
                    -100,
                    -100,
                    100,
                    100,
                    100,
                ],
                type='PointsRangeFilter'),
        ],
        type='MultiScaleFlipAug3D'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'pts_semantic_mask',
        ],
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'lidar2img',
            'depth2img',
            'cam2img',
            'pad_shape',
            'scale_factor',
            'flip',
            'pcd_horizontal_flip',
            'pcd_vertical_flip',
            'box_mode_3d',
            'box_type_3d',
            'img_norm_cfg',
            'num_pts_feats',
            'pcd_trans',
            'sample_idx',
            'pcd_scale_factor',
            'pcd_rotation',
            'pcd_rotation_angle',
            'lidar_path',
            'transformation_3d_flow',
            'trans_mat',
            'affine_aug',
            'sweep_img_metas',
            'ori_cam2img',
            'cam2global',
            'crop_offset',
            'img_crop_offset',
            'resize_img_shape',
            'lidar2cam',
            'ori_lidar2img',
            'num_ref_frames',
            'num_views',
            'ego2global',
            'axis_align_matrix',
            'one_hot_vector',
        ),
        type='Pack3DDetInputs'),
]
input_modality = dict(use_camera=False, use_lidar=True)
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.0005
metainfo = dict(classes=[
    'Pedestrian',
    'Cyclist',
    'Car',
])
model = dict(
    anchors=[
        [
            0.8,
            0.6,
            1.73,
        ],
        [
            1.76,
            0.6,
            1.73,
        ],
        [
            3.9,
            1.6,
            1.56,
        ],
    ],
    box_estimation_backbone=dict(
        fp_channels=(),
        in_channels=5,
        norm_cfg=dict(type='BN2d'),
        num_points=(
            32,
            16,
            None,
        ),
        num_samples=(
            32,
            16,
            None,
        ),
        radius=(
            0.2,
            0.4,
            None,
        ),
        sa_cfg=dict(
            normalize_xyz=False,
            pool_mod='max',
            type='PointSAModule',
            use_xyz=True),
        sa_channels=[
            [
                64,
                64,
                128,
            ],
            [
                128,
                128,
                256,
            ],
            [
                256,
                256,
                512,
            ],
        ],
        type='PointNet2SASSG'),
    corner_loss=True,
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    fc_channels=[
        512,
        256,
    ],
    fc_input_channels=515,
    gamma_corner=0.1,
    loss_anchors_reg=dict(
        beta=0.1111111111111111, loss_weight=5.0, type='mmdet.SmoothL1Loss'),
    loss_center_bbox=dict(
        beta=0.1111111111111111, loss_weight=1.0, type='mmdet.SmoothL1Loss'),
    loss_center_tnet=dict(
        beta=0.1111111111111111, loss_weight=5.0, type='mmdet.SmoothL1Loss'),
    loss_cls_anchors=dict(
        loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
    loss_cls_headings=dict(
        loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
    loss_headings_reg=dict(
        beta=0.1111111111111111, loss_weight=10.0, type='mmdet.SmoothL1Loss'),
    max_num_points=384,
    one_hot_len=3,
    rotations=[
        -3.14,
        -2.617,
        -2.093,
        -1.57,
        -1.046,
        -0.523,
        0,
        0.523,
        1.046,
        1.57,
        2.093,
        2.617,
    ],
    seg_backbone=dict(
        aggregation_channels=(
            None,
            None,
            None,
        ),
        dilated_group=(
            False,
            False,
            False,
        ),
        fps_mods=(
            'D-FPS',
            'D-FPS',
            'D-FPS',
        ),
        fps_sample_range_lists=(
            -1,
            -1,
            -1,
        ),
        in_channels=5,
        norm_cfg=dict(type='BN2d'),
        num_points=(
            128,
            32,
            None,
        ),
        num_samples=[
            [
                32,
                64,
                128,
            ],
            [
                64,
                64,
                128,
            ],
            [
                None,
            ],
        ],
        out_indices=(
            0,
            1,
            2,
        ),
        radii=[
            [
                0.2,
                0.4,
                0.8,
            ],
            [
                0.4,
                0.8,
                1.6,
            ],
            [
                None,
            ],
        ],
        sa_cfg=dict(
            normalize_xyz=False,
            pool_mod='max',
            type='PointSAModuleMSG',
            use_xyz=True),
        sa_channels=[
            [
                [
                    32,
                    32,
                    64,
                ],
                [
                    64,
                    64,
                    128,
                ],
                [
                    64,
                    96,
                    128,
                ],
            ],
            [
                [
                    64,
                    64,
                    128,
                ],
                [
                    128,
                    128,
                    256,
                ],
                [
                    128,
                    128,
                    256,
                ],
            ],
            [
                [
                    128,
                    256,
                    1024,
                ],
            ],
        ],
        type='PointNet2SAMSG'),
    seg_decode_head=dict(
        act_cfg=dict(type='ReLU'),
        channels=128,
        conv_cfg=dict(type='Conv1d'),
        dropout_ratio=0.3,
        fp_channels=(
            (
                1667,
                128,
                128,
            ),
            (
                448,
                128,
                128,
            ),
            (
                128,
                128,
                128,
            ),
        ),
        loss_decode=dict(
            class_weight=None,
            loss_weight=1.0,
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(type='BN1d'),
        num_classes=2,
        type='PointNet2Head'),
    test_cfg=dict(sample_seg_points=64),
    train_cfg=dict(sample_seg_points=64),
    type='FrustumPointNet',
    with_one_hot=True)
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(
        betas=(
            0.95,
            0.99,
        ), lr=0.0005, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(
        factor=0.5,
        min_value=1e-05,
        monitor='IoU_3D',
        patience=5,
        rule='greater',
        type='ReduceOnPlateauLR'),
]
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=data_root + 'kitti_frustum_info_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromDict',
                use_dim=5),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            -100,
                            -100,
                            -100,
                            100,
                            100,
                            100,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(
                keys=[
                    'points',
                ],
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'lidar2img',
                    'depth2img',
                    'cam2img',
                    'pad_shape',
                    'scale_factor',
                    'flip',
                    'pcd_horizontal_flip',
                    'pcd_vertical_flip',
                    'box_mode_3d',
                    'box_type_3d',
                    'img_norm_cfg',
                    'num_pts_feats',
                    'pcd_trans',
                    'sample_idx',
                    'pcd_scale_factor',
                    'pcd_rotation',
                    'pcd_rotation_angle',
                    'lidar_path',
                    'transformation_3d_flow',
                    'trans_mat',
                    'affine_aug',
                    'sweep_img_metas',
                    'ori_cam2img',
                    'cam2global',
                    'crop_offset',
                    'img_crop_offset',
                    'resize_img_shape',
                    'lidar2cam',
                    'ori_lidar2img',
                    'num_ref_frames',
                    'num_views',
                    'ego2global',
                    'axis_align_matrix',
                    'one_hot_vector',
                ),
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='FrustumDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    box_type_3d='LiDAR',
    classes=[
        'Pedestrian',
        'Cyclist',
        'Car',
    ],
    iou_thresholds=[
        0.5,
        0.5,
        0.7,
    ],
    type='FrustumLocalizationMetric',
    with_cls_out=False)
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromDict',
        use_dim=5),
    dict(
        flip=False,
        img_scale=(
            1333,
            800,
        ),
        pts_scale_ratio=1,
        transforms=[
            dict(
                rot_range=[
                    0,
                    0,
                ],
                scale_ratio_range=[
                    1.0,
                    1.0,
                ],
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTrans'),
            dict(type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    -100,
                    -100,
                    -100,
                    100,
                    100,
                    100,
                ],
                type='PointsRangeFilter'),
        ],
        type='MultiScaleFlipAug3D'),
    dict(
        keys=[
            'points',
        ],
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'lidar2img',
            'depth2img',
            'cam2img',
            'pad_shape',
            'scale_factor',
            'flip',
            'pcd_horizontal_flip',
            'pcd_vertical_flip',
            'box_mode_3d',
            'box_type_3d',
            'img_norm_cfg',
            'num_pts_feats',
            'pcd_trans',
            'sample_idx',
            'pcd_scale_factor',
            'pcd_rotation',
            'pcd_rotation_angle',
            'lidar_path',
            'transformation_3d_flow',
            'trans_mat',
            'affine_aug',
            'sweep_img_metas',
            'ori_cam2img',
            'cam2global',
            'crop_offset',
            'img_crop_offset',
            'resize_img_shape',
            'lidar2cam',
            'ori_lidar2img',
            'num_ref_frames',
            'num_views',
            'ego2global',
            'axis_align_matrix',
            'one_hot_vector',
        ),
        type='Pack3DDetInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=2)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        dataset=dict(
            ann_file=data_root + 'kitti_frustum_info_train.pkl',
            backend_args=None,
            box_type_3d='LiDAR',
            metainfo=dict(classes=[
                'Pedestrian',
                'Cyclist',
                'Car',
            ]),
            pipeline=[
                dict(
                    backend_args=None,
                    coord_type='LIDAR',
                    load_dim=5,
                    type='LoadPointsFromDict',
                    use_dim=5),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    global_rot_range=[
                        0.0,
                        0.0,
                    ],
                    num_try=100,
                    rot_range=[
                        -0.15707963267,
                        0.15707963267,
                    ],
                    translation_std=[
                        0.1,
                        0.1,
                        0.01,
                    ],
                    type='ObjectNoise'),
                dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
                dict(
                    rot_range=[
                        -0.0,
                        0.0,
                    ],
                    scale_ratio_range=[
                        0.95,
                        1.05,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    point_cloud_range=[
                        -100,
                        -100,
                        -100,
                        100,
                        100,
                        100,
                    ],
                    type='PointsRangeFilter'),
                dict(
                    point_cloud_range=[
                        -100,
                        -100,
                        -100,
                        100,
                        100,
                        100,
                    ],
                    type='ObjectRangeFilter'),
                dict(type='PointShuffle'),
                dict(
                    keys=[
                        'points',
                        'gt_bboxes_3d',
                        'gt_labels_3d',
                        'pts_semantic_mask',
                    ],
                    meta_keys=(
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'lidar2img',
                        'depth2img',
                        'cam2img',
                        'pad_shape',
                        'scale_factor',
                        'flip',
                        'pcd_horizontal_flip',
                        'pcd_vertical_flip',
                        'box_mode_3d',
                        'box_type_3d',
                        'img_norm_cfg',
                        'num_pts_feats',
                        'pcd_trans',
                        'sample_idx',
                        'pcd_scale_factor',
                        'pcd_rotation',
                        'pcd_rotation_angle',
                        'lidar_path',
                        'transformation_3d_flow',
                        'trans_mat',
                        'affine_aug',
                        'sweep_img_metas',
                        'ori_cam2img',
                        'cam2global',
                        'crop_offset',
                        'img_crop_offset',
                        'resize_img_shape',
                        'lidar2cam',
                        'ori_lidar2img',
                        'num_ref_frames',
                        'num_views',
                        'ego2global',
                        'axis_align_matrix',
                        'one_hot_vector',
                    ),
                    type='Pack3DDetInputs'),
            ],
            test_mode=False,
            type='FrustumDataset'),
        times=1,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromDict',
        use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        global_rot_range=[
            0.0,
            0.0,
        ],
        num_try=100,
        rot_range=[
            -0.15707963267,
            0.15707963267,
        ],
        translation_std=[
            0.1,
            0.1,
            0.01,
        ],
        type='ObjectNoise'),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(
        rot_range=[
            -0.0,
            0.0,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        type='GlobalRotScaleTrans'),
    dict(
        point_cloud_range=[
            -100,
            -100,
            -100,
            100,
            100,
            100,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            -100,
            -100,
            -100,
            100,
            100,
            100,
        ],
        type='ObjectRangeFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'pts_semantic_mask',
        ],
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'lidar2img',
            'depth2img',
            'cam2img',
            'pad_shape',
            'scale_factor',
            'flip',
            'pcd_horizontal_flip',
            'pcd_vertical_flip',
            'box_mode_3d',
            'box_type_3d',
            'img_norm_cfg',
            'num_pts_feats',
            'pcd_trans',
            'sample_idx',
            'pcd_scale_factor',
            'pcd_rotation',
            'pcd_rotation_angle',
            'lidar_path',
            'transformation_3d_flow',
            'trans_mat',
            'affine_aug',
            'sweep_img_metas',
            'ori_cam2img',
            'cam2global',
            'crop_offset',
            'img_crop_offset',
            'resize_img_shape',
            'lidar2cam',
            'ori_lidar2img',
            'num_ref_frames',
            'num_views',
            'ego2global',
            'axis_align_matrix',
            'one_hot_vector',
        ),
        type='Pack3DDetInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=data_root + 'kitti_frustum_info_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        load_eval_anns=True,
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromDict',
                use_dim=5),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            -100,
                            -100,
                            -100,
                            100,
                            100,
                            100,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(
                keys=[
                    'points',
                ],
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'lidar2img',
                    'depth2img',
                    'cam2img',
                    'pad_shape',
                    'scale_factor',
                    'flip',
                    'pcd_horizontal_flip',
                    'pcd_vertical_flip',
                    'box_mode_3d',
                    'box_type_3d',
                    'img_norm_cfg',
                    'num_pts_feats',
                    'pcd_trans',
                    'sample_idx',
                    'pcd_scale_factor',
                    'pcd_rotation',
                    'pcd_rotation_angle',
                    'lidar_path',
                    'transformation_3d_flow',
                    'trans_mat',
                    'affine_aug',
                    'sweep_img_metas',
                    'ori_cam2img',
                    'cam2global',
                    'crop_offset',
                    'img_crop_offset',
                    'resize_img_shape',
                    'lidar2cam',
                    'ori_lidar2img',
                    'num_ref_frames',
                    'num_views',
                    'ego2global',
                    'axis_align_matrix',
                    'one_hot_vector',
                ),
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='FrustumDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    box_type_3d='LiDAR',
    classes=[
        'Pedestrian',
        'Cyclist',
        'Car',
    ],
    iou_thresholds=[
        0.5,
        0.5,
        0.7,
    ],
    type='FrustumLocalizationMetric',
    with_cls_out=False)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])