# model settings
model = dict(
    type='AFSingleAPERoITransformer',
    pretrained='/disk2/zzr/resnet50.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='PAFPN_V2',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    rpn_head=dict(
        type='ATSSHead',
        num_classes=17,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64, 128],
        anchor_base_sizes=None,
        target_means=(.0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2),
        train_cfg=dict(
            assigner=dict(
                type='ATSSAssigner',
                topk=9),
            smoothl1_beta=0.11,
            gamma=2.0,
            alpha=0.25,
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    bbox_roi_extractor=dict(
        type='MultiRoIExtractor',
        panet_channels=1024,
        # type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[8, 16, 32, 64, 128]),
    bbox_head=dict(
        type='FusedAPEFCBBoxHeadRbbox',
        # type='FusedFCBBoxHeadRbbox',
        # type='SharedFCBBoxHeadRbbox',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=17,
        target_means=[0., 0., 0., 0., 0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],
        reg_class_agnostic=True,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
    )
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssignerCy',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    # rpn_proposal=dict(
    #     nms_across_levels=False,
    #     nms_pre=2000,
    #     nms_post=2000,
    #     max_num=2000,
    #     nms_thr=0.7,
    #     min_bbox_size=0),
    rpn_proposal=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.0,
        nms=dict(type='nms', iou_thr=0.7),
        max_per_img=2000),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssignerCy',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssignerRbbox',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomRbboxSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ])
test_cfg = dict(
    # rpn=dict(
    #     # TODO: test nms 2000
    #     nms_across_levels=False,
    #     nms_pre=2000,
    #     nms_post=2000,
    #     max_num=2000,
    #     nms_thr=0.7,
    #     min_bbox_size=0),
    rpn=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.0,
        nms=dict(type='nms', iou_thr=0.7),
        max_per_img=2000),
    rcnn=dict(
        # score_thr=0.05, nms=dict(type='py_cpu_nms_poly_fast', iou_thr=0.1), max_per_img=1000)
        score_thr = 0.05, nms = dict(type='py_cpu_nms_poly_fast', iou_thr=0.1), max_per_img = 2000)
# soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'DOTA1_5Dataset_v3'
data_root = '/disk2/zzr/dataset_dota1_5/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval1024_obb/DOTA1_5_trainval1024.json',
        img_prefix=data_root + 'trainval1024_obb/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True,
        rotate_aug=dict(border_value=0, small_filter=6)
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval1024_obb/DOTA1_5_trainval1024.json',
        img_prefix=data_root + 'trainval1024_obb/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test1024_obb/DOTA1_5_test1024.json',
        img_prefix=data_root + 'test1024_obb/images',
        # ann_file = data_root + 'test1024_ms/DOTA1_5_test1024_allms.json',
        # img_prefix = data_root + 'test1024_ms/images',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.00125/2*2, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/disk2/zzr/work_dirs/RoITrans_r50_pafpn_V2_atss_head_APE_RCNN_1x_dota1_5'
load_from = None
resume_from = None
workflow = [('train', 1)]