import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *
class air_101_rcnn_dcn(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = [3, 4, 22, 3] # use for 101
        self.filter_list = [64,256, 512, 1024, 2048]
        self.bn_global_ = True
        self.bn_mom = 0.9


    def residual_unit(self,data, num_filter, stride, dim_match, name,dilate = 1, bottle_neck=True, bn_mom=0.9, workspace=512,
                      bn_global=True,deform_conv = False):
        """Return resnext Unit symbol for building resnext
        Parameters
        ----------
        data : str
            Input data
        num_filter : int
            Number of output channels
        bnf : int
            Bottle neck channels factor with regard to num_filter
        stride : tupe
            Stride used in convolution
        dim_match : Boolen
            True means channel number between input and output is the same, otherwise means differ
        name : str
            Base name of the operators
        workspace : int
            Workspace used in convolution operator
        """
        if bottle_neck:

            # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
            conv1_1 = mx.sym.Convolution(data=data, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=stride,
                                       pad=(0, 0),
                                       no_bias=True, workspace=workspace, name=name + '_conv1_1')
            conv1_1_bn = mx.sym.BatchNorm(data=conv1_1, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=bn_global,
                                   name=name + '_conv1_1_bn')
            conv1_1_relu = mx.sym.Activation(data=conv1_1_bn, act_type='relu', name=name + '_conv1_1_relu')
            if deform_conv:

                conv1_2_offset = mx.symbol.Convolution(name=name+'_conv1_2_offset', data=conv1_1_relu,
                                                     num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1),
                                                     dilate=(2, 2), cudnn_off=True)
                conv1_2 = mx.contrib.symbol.DeformableConvolution(name=name+'_conv1_2',
                                                                data=conv1_1_relu,
                                                                offset=conv1_2_offset,
                                                                num_filter=int(num_filter * 0.25), pad=(dilate, dilate), kernel=(3, 3),
                                                                num_deformable_group=4,num_group = 1,
                                                                stride=(1, 1), dilate=(dilate, dilate), no_bias=True,workspace = workspace)

            else:
                conv1_2 = mx.sym.Convolution(data=conv1_1_relu, num_filter=int(num_filter * 0.25), kernel=(3, 3), \
                                         stride=(1, 1),dilate = (dilate,dilate),
                                         pad=(dilate, dilate),
                                         no_bias=True, workspace=workspace, name=name + '_conv1_2')


            conv2_1 = mx.sym.Convolution(data=data, num_filter=int(num_filter * 0.125), kernel=(1, 1), stride=(1, 1),
                                                                      pad=(0, 0),
                                                                      no_bias=True, workspace=workspace, name=name + '_conv2_1')
            conv2_1_bn = mx.sym.BatchNorm(data=conv2_1, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=bn_global,
                                          name=name + '_conv2_1_bn')
            conv2_1_relu = mx.sym.Activation(data=conv2_1_bn, act_type='relu', name=name + '_conv2_1_relu')
            if deform_conv:

                conv2_2_offset = mx.symbol.Convolution(name=name+'_conv2_2_offset', data=conv2_1_relu,
                                                       num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1),
                                                       dilate=(2, 2), cudnn_off=True)
                conv2_2 = mx.contrib.symbol.DeformableConvolution(name=name+'_conv2_2',
                                                                  data=conv2_1_relu,
                                                                  offset=conv2_2_offset,
                                                                  num_filter=int(num_filter * 0.125), pad=(dilate, dilate), kernel=(3, 3),
                                                                  num_deformable_group=4,num_group = 1,
                                                                  stride=stride, dilate=(dilate, dilate), no_bias=True,workspace = workspace)

            else:
                conv2_2 = mx.sym.Convolution(data=conv2_1_relu, num_filter=int(num_filter * 0.125), kernel=(3, 3), stride=stride,
                                         pad=(dilate, dilate),dilate=(dilate,dilate),
                                         no_bias=True, workspace=workspace, name=name + '_conv2_2')
            conv2_2_bn = mx.sym.BatchNorm(data=conv2_2, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=bn_global,
                                          name=name + '_conv2_2_bn')
            conv2_2_relu = mx.sym.Activation(data=conv2_2_bn, act_type='relu', name=name + '_conv2_2_relu')


            if deform_conv:

                conv2_3_offset = mx.symbol.Convolution(name=name+'_conv2_3_offset', data=conv2_2_relu,
                                                       num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1),
                                                       dilate=(2, 2), cudnn_off=True)
                conv2_3 = mx.contrib.symbol.DeformableConvolution(name=name+'_conv2_3',
                                                                  data=conv2_2_relu,
                                                                  offset=conv2_3_offset,
                                                                  num_filter=int(num_filter * 0.125), pad=(2, 2), kernel=(3, 3),
                                                                  num_deformable_group=4,num_group = 1,
                                                                  stride=(1, 1), dilate=(2, 2), no_bias=True,workspace = workspace)
            else:
                conv2_3 = mx.sym.Convolution(data=conv2_2_relu, num_filter=int(num_filter * 0.125), kernel=(3, 3), stride=(1, 1),
                                         pad=(dilate, dilate),dilate = (dilate,dilate),
                                         no_bias=True, workspace=workspace, name=name + '_conv2_3')
            cat = mx.sym.concat(conv1_2,conv2_3,name = name+'_cat')
            cat_bn = mx.sym.BatchNorm(data=cat, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=bn_global,
                                          name=name + '_cat_bn')
            cat_relu = mx.sym.Activation(data=cat_bn, act_type='relu', name=name + '_cat_relu')
            conv3 = mx.sym.Convolution(data=cat_relu, num_filter=int(num_filter), kernel=(1, 1), stride=(1, 1),
                                         pad=(0, 0),
                                         no_bias=True, workspace=workspace, name=name + '_conv3')
            conv3_bn = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=bn_global,
                                          name=name + '_conv3_bn')
            if dim_match:
                shortcut = data
            else:

                match_conv = mx.sym.Convolution(data=data, num_filter=int(num_filter), kernel=(1, 1), stride=stride,
                                       pad=(0, 0),
                                       no_bias=True, workspace=workspace, name=name + '_match_conv')
                match_conv_bn = mx.sym.BatchNorm(data=match_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=bn_global,
                                        name=name + '_match_conv_bn')
                shortcut = match_conv_bn
            eltwise = conv3_bn + shortcut

            '''
            if dilate and not deform_conv:
                stride = (1, 1)
                conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.5), num_group=32, kernel=(3, 3),
                                           stride=stride, pad=(2, 2), dilate=(2, 2),
                                           no_bias=True, workspace=workspace, name=name + '_conv2')
            elif deform_conv and dilate:
                stride = (1,1)
                conv2_offset = mx.symbol.Convolution(name=name+'_conv2_offset', data=act1,
                                                              num_filter=72, pad=(2, 2), kernel=(3, 3), stride=(1, 1),
                                                              dilate=(2, 2), cudnn_off=True)
                conv2 = mx.contrib.symbol.DeformableConvolution(name=name+'_conv2',
                                                                         data=act1,
                                                                         offset=conv2_offset,
                                                                         num_filter=int(num_filter * 0.5), pad=(2, 2), kernel=(3, 3),
                                                                         num_deformable_group=4,num_group = 32,
                                                                         stride=(1, 1), dilate=(2, 2), no_bias=True,workspace = workspace)

            else:
                conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.5), num_group=32, kernel=(3, 3),
                                           stride=stride, pad=(1, 1),
                                           no_bias=True, workspace=workspace, name=name + '_conv2')

            bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=bn_global,
                                   name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')

            conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                       no_bias=True,
                                       workspace=workspace, name=name + '_conv3')
            bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=bn_global,
                                   name=name + '_bn3')

            if dim_match:
                shortcut = data
            else:
                stride = (2,2)
                shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                                   no_bias=True,
                                                   workspace=workspace, name=name + '_sc')
                shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                                            use_global_stats=bn_global, name=name + '_sc_bn')
            eltwise = bn3 + shortcut
            return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')
        else:
            conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                       no_bias=True, workspace=workspace, name=name + '_conv1')
            bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, use_global_stats=bn_global,
                                   name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

            conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                       no_bias=True, workspace=workspace, name=name + '_conv2')
            bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, use_global_stats=bn_global,
                                   name=name + '_bn2')

            if dim_match:
                shortcut = data
            else:
                shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                                   no_bias=True,
                                                   workspace=workspace, name=name + '_sc')
                shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                                            use_global_stats=bn_global, name=name + '_sc_bn')

            eltwise = bn2 + shortcut
            '''
            return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')

    def get_rpn(self, conv_feat, num_anchors):
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
        return rpn_cls_score, rpn_bbox_pred
    def get_air_101_conv4(self, data, num_stage = 3, bottle_neck = True):
        body = mx.sym.Convolution(data=data, num_filter=self.filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                  no_bias=True, name="res1", workspace=self.workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=self.bn_mom, use_global_stats=self.bn_global_,
                                name='res1_bn')
        body = mx.sym.Activation(data=body, act_type='relu', name='res1_relu')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
        name_idex = 2
        for i in range(num_stage):
            print(name_idex)
            bn_global_ = self.bn_global_
            body = self.residual_unit(body, self.filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False, dilate=1,
                                 name='res'+str(name_idex), bottle_neck=bottle_neck, workspace=self.workspace,
                                 bn_global=bn_global_)
            name_idex = name_idex+1
            for j in range(self.units[i] - 1):
                body = self.residual_unit(data = body, num_filter=self.filter_list[i + 1], stride = (1, 1), dim_match=True, name='res'+str(name_idex),
                                     bottle_neck=bottle_neck, workspace=self.workspace, bn_global=bn_global_,dilate=1)
                name_idex = name_idex+1

        print(name_idex)
        return body
    def get_air_101_conv5(self, conv_fea):
        i = 3
        name_idex = 32
        body = self.residual_unit(conv_fea, self.filter_list[i + 1], stride = (1,1), dim_match=False, dilate=2,
                                  name='res'+str(name_idex), bottle_neck=True, workspace=self.workspace,
                                  bn_global=self.bn_global_,
                                  deform_conv=True)
        name_idex = name_idex+1
        for j in range(self.units[i] - 1):
            body = self.residual_unit(body, self.filter_list[i + 1], (1, 1), True, name='res'+str(name_idex),
                                      bottle_neck=True, workspace=self.workspace, bn_global=self.bn_global_,dilate=2, \
                                      deform_conv=True)
            name_idex = name_idex+1
        return body


    def get_symbol(self,cfg,is_train = True):
        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")
            gt_boxes = mx.sym.Variable(name="gt_boxes")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        conv_feat = self.get_air_101_conv4(data)
        # res5
        relu1 = self.get_air_101_conv5(conv_fea=conv_feat)

        rpn_cls_score, rpn_bbox_pred = self.get_rpn(conv_feat, num_anchors)

        if is_train:
            # prepare rpn data
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                normalization='valid', use_ignore=True, ignore_label=-1,
                                                name="rpn_cls_prob")

            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0,
                                                                data=(rpn_bbox_pred - rpn_bbox_target))

            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                            grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            # ROI proposal
            rpn_cls_act = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
            rpn_cls_act_reshape = mx.sym.Reshape(
                data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
            if cfg.TRAIN.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            # ROI proposal target
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
            rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)
        else:
            # ROI Proposal
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
            if cfg.TEST.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

        conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=256, name="conv_new_1")
        conv_new_1_relu = mx.sym.Activation(data=conv_new_1, act_type='relu', name='conv_new_1_relu')
        offset_t = mx.contrib.sym.DeformablePSROIPooling(name='offset_t', data=conv_new_1_relu, rois=rois, group_size=1,
                                                         pooled_size=7,
                                                         sample_per_part=4, no_trans=True, part_size=7, output_dim=256,
                                                         spatial_scale=0.0625)
        offset = mx.sym.FullyConnected(name='offset', data=offset_t, num_hidden=7 * 7 * 2, lr_mult=0.01)
        offset_reshape = mx.sym.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")

        deformable_roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool', data=conv_new_1_relu,
                                                                    rois=rois,
                                                                    trans=offset_reshape, group_size=1, pooled_size=7,
                                                                    sample_per_part=4,
                                                                    no_trans=False, part_size=7, output_dim=256,
                                                                    spatial_scale=0.0625, trans_std=0.1)

        # 2 fc
        fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=deformable_roi_pool, num_hidden=1024)
        fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')

        fc_new_2 = mx.symbol.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=1024)
        fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')

        # cls_score/bbox_pred
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4)

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes,
                                                               roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem,
                                                normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                                  data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                            grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid')
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                            data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_loss_reshape')
            group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred])

        self.sym = group
        return group

    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        arg_params['res32_conv1_2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res32_conv1_2_offset_weight'])
        arg_params['res32_conv1_2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res32_conv1_2_offset_bias'])
        arg_params['res32_conv2_2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res32_conv2_2_offset_weight'])
        arg_params['res32_conv2_2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res32_conv2_2_offset_bias'])
        arg_params['res32_conv2_3_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res32_conv2_3_offset_weight'])
        arg_params['res32_conv2_3_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res32_conv2_3_offset_bias'])

        arg_params['res33_conv1_2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res33_conv1_2_offset_weight'])
        arg_params['res33_conv1_2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res33_conv1_2_offset_bias'])
        arg_params['res33_conv2_2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res33_conv2_2_offset_weight'])
        arg_params['res33_conv2_2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res33_conv2_2_offset_bias'])
        arg_params['res33_conv2_3_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res33_conv2_3_offset_weight'])
        arg_params['res33_conv2_3_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res33_conv2_3_offset_bias'])

        arg_params['res34_conv1_2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res34_conv1_2_offset_weight'])
        arg_params['res34_conv1_2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res34_conv1_2_offset_bias'])
        arg_params['res34_conv2_2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res34_conv2_2_offset_weight'])
        arg_params['res34_conv2_2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res34_conv2_2_offset_bias'])
        arg_params['res34_conv2_3_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res34_conv2_3_offset_weight'])
        arg_params['res34_conv2_3_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res34_conv2_3_offset_bias'])

        arg_params['offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['offset_weight'])
        arg_params['offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['offset_bias'])

        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['fc_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1_weight'])
        arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])
        arg_params['fc_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2_weight'])
        arg_params['fc_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_2_bias'])
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])

    def init_weight_rpn(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_rpn(cfg, arg_params, aux_params)
        self.init_weight_rcnn(cfg, arg_params, aux_params)
