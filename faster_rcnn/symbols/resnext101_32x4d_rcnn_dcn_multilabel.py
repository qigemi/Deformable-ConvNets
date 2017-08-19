import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *
import numpy as np
class CrossEntropyLoss(mx.operator.CustomOp):
    """An output layer that calculates gradient for cross-entropy loss
    y * log(p) + (1-y) * log(p)
    for label "y" and prediction "p".
    However, the output of this layer is the original prediction -- same as
    the "data" input, making it useful for tasks like "predict".
    If you actually want to use the calculated loss, see CrossEntropyLoss op.
    This is useful for multi-label prediction where each possible output
    label is considered independently.
    Cross-entropy loss provides a very large penalty for guessing
    the wrong answer (0 or 1) confidently.
    The gradient calculation is optimized for y only being 0 or 1.
    """

    eps = 1e-6 # Avoid -inf when taking log(0)
    eps1 = 1. + eps
    eps_1 = 1. - eps

    def forward(self, is_train, req, in_data, out_data, aux):
        # Shapes:
        #  b = minibatch size
        #  d = number of dimensions
        actually_calculate_loss = True
        if actually_calculate_loss:
            p = in_data[0].asnumpy()  # shape=(b,d)
            y = in_data[1].asnumpy()
            out = y * np.log(p+self.eps) + (1.-y) * np.log((self.eps1) - p)
            out = -out
            self.assign(out_data[0], req[0], mx.nd.array(out))
        else:
            # Just copy the predictions forward
            self.assign(out_data[0], req[0], in_data[0])


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        #self.approx_backward(req, out_grad, in_data, out_data, in_grad, aux)
        self.exact_backward(req, out_grad, in_data, out_data, in_grad, aux)

    def approx_backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """Correct grad = (y-p)/(p-p^2)
        But if y is just 1 or 0, then this simplifies to
        grad = 1/(p-1+y)
        which is more numerically stable
        """
        p = in_data[0].asnumpy()  # shape=(b,d)
        y = in_data[1].asnumpy()
        d_new = p - self.eps_1 + y
        d_new[d_new==0] = self.eps_1
        grad = -1. / d_new
        self.assign(in_grad[0], req[0], mx.nd.array(grad))


    def exact_backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """grad = (y-p)/(p-p^2)
        """
        p = in_data[0].asnumpy()  # shape=(b,d)
        y = in_data[1].asnumpy()  # seems right
        grad = (p - y) / ((p+self.eps) * (self.eps1 - p))
        self.assign(in_grad[0], req[0], mx.nd.array(grad))


@mx.operator.register("CrossEntropyLoss")
class CrossEntropyProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CrossEntropyProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data','label']

    def list_outputs(self):
        return ['preds']

    def create_operator(self, ctx, shapes, dtypes):
        return CrossEntropyLoss()

    def infer_shape(self, in_shape):
        if in_shape[0] != in_shape[1]:
            raise ValueError("Input shapes differ. data:%s. label:%s. must be same"
                    % (str(in_shape[0]),str(in_shape[1])))
        output_shape = in_shape[0]
        return in_shape, [output_shape], []
class resnext101_32x4d_rcnn_dcn_multilabel(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = [3, 4, 23, 3] # use for 101
        self.filter_list = [64,256, 512, 1024, 2048]
        self.bn_global_ = True
        self.bn_mom = 0.9


    def residual_unit(self,data, num_filter, stride, dim_match, dilate, name, bottle_neck=True, bn_mom=0.9, workspace=512,
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
            conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter * 0.5), kernel=(1, 1), stride=(1, 1),
                                       pad=(0, 0),
                                       no_bias=True, workspace=workspace, name=name + '_conv1')
            bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=bn_global,
                                   name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

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
    def get_resnext101_32x4d_conv4(self,data,num_stage = 3,bottle_neck = True):
        data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=self.bn_mom, use_global_stats=self.bn_global_,
                                name='bn_data')
        body = mx.sym.Convolution(data=data, num_filter=self.filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=self.workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=self.bn_mom, use_global_stats=self.bn_global_,
                                name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
        for i in range(num_stage):
            bn_global_ = self.bn_global_ # after roi-pooling, do not use use_global_stats
            dilate = False
            body = self.residual_unit(body, self.filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False, dilate=dilate,
                                 name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=self.workspace,
                                 bn_global=bn_global_)
            for j in range(self.units[i] - 1):
                body = self.residual_unit(data = body, num_filter=self.filter_list[i + 1], stride = (1, 1), dim_match=True, name='stage%d_unit%d' % (i + 1, j + 2),
                                     bottle_neck=bottle_neck, workspace=self.workspace, bn_global=bn_global_,dilate=False)
        return body
    def get_resnext101_32x4d_conv5(self,conv_fea):
        i = 3
        dilate = True
        body = self.residual_unit(conv_fea, self.filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False, dilate=dilate,
                                  name='stage%d_unit%d' % (i + 1, 1), bottle_neck=True, workspace=self.workspace,
                                  bn_global=self.bn_global_)
        for j in range(self.units[i] - 1):
            body = self.residual_unit(body, self.filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                      bottle_neck=True, workspace=self.workspace, bn_global=self.bn_global_,dilate=True,deform_conv=True)
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
            multi_label = mx.sym.Variable(name='multi_label')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        conv_feat = self.get_resnext101_32x4d_conv4(data)
        #rpn_fea = mx.sym.Convolution(data=conv_feat, kernel=(1, 1), num_filter=num_classes, name="rpn_fea")
        #rpn_relu = mx.sym.Activation(data=rpn_fea, act_type='relu', name='rpn_relu')
        # res5
        relu1 = self.get_resnext101_32x4d_conv5(conv_fea=conv_feat)
        rpn_cls_score, rpn_bbox_pred = self.get_rpn(conv_feat, num_anchors)

        if is_train:
            #muli_label
            multi_label_weight = 0.01
            rpn_pool = mx.symbol.Pooling(name='rpn_pool', data=conv_feat, global_pool=True,kernel = (7,7), pool_type='avg')
            rpn_flat = mx.sym.Flatten(data=rpn_pool)
            rpn_fc = mx.symbol.FullyConnected(name='rpn_fc', data=rpn_flat, num_hidden=num_classes)
            rpn_sig = mx.sym.Activation(data=rpn_fc, name='rpn_sig',act_type  = 'sigmoid')
            multi_label_loss_ = multi_label_weight*mx.symbol.Custom(data=rpn_sig,label = multi_label, name='multi_label_loss_', op_type='CrossEntropyLoss')
            multi_label_loss = mx.sym.MakeLoss(data=multi_label_loss_,name='multi_label_loss',grad_scale = 1)
            # prepare rpn data
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                   normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")

            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))

            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

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
        #deform psroi
        offset_t = mx.contrib.sym.DeformablePSROIPooling(name='offset_t', data=conv_new_1_relu, rois=rois, group_size=1, pooled_size=7,
                                                         sample_per_part=4, no_trans=True, part_size=7, output_dim=256, spatial_scale=0.0625)
        offset = mx.sym.FullyConnected(name='offset', data=offset_t, num_hidden=7 * 7 * 2, lr_mult=0.01)
        offset_reshape = mx.sym.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")

        deformable_roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool', data=conv_new_1_relu, rois=rois,
                                                                    trans=offset_reshape, group_size=1, pooled_size=7, sample_per_part=4,
                                                                    no_trans=False, part_size=7, output_dim=256, spatial_scale=0.0625, trans_std=0.1)

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
            group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label),multi_label_loss,mx.sym.BlockGrad(rpn_sig)])
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
        arg_params['stage4_unit2_conv2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit2_conv2_offset_weight'])
        arg_params['stage4_unit2_conv2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit2_conv2_offset_bias'])
        arg_params['stage4_unit3_conv2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit3_conv2_offset_weight'])
        arg_params['stage4_unit3_conv2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit3_conv2_offset_bias'])
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        #arg_params['rpn_fea_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_fea_weight'])
        #arg_params['rpn_fea_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_fea_bias'])
        arg_params['rpn_fc_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_fc_weight'])
        arg_params['rpn_fc_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_fc_bias'])
        arg_params['offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['offset_weight'])
        arg_params['offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['offset_bias'])
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
