"""ROI Global Context Operator enlarges the rois with its surrounding areas, to provide contextual information
"""

from __future__ import print_function
import mxnet as mx

DEBUG = False


class ROIGlobalContextOperator(mx.operator.CustomOp):
    def __init__(self, global_context_scale):
        super(ROIGlobalContextOperator, self).__init__()
        self._global_context_scale = global_context_scale


    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].copy()  # rois=[cls, x1, y1, x2, y2]
        im_info = in_data[1].copy()   # im_info=(height,width)
        #im_info = mx.ndarray.slice_axis(im_info, axis=0, begin=0, end=1)
        if DEBUG:
            print('im_info:',im_info.asnumpy())
        y = out_data[0]
        for idx, roi in enumerate(x):
            roi_ctr_x = 0.5 * (roi[1] + roi[3])  # (x1+x2)/2
            roi_ctr_y = 0.5 * (roi[2] + roi[4])  # (y1+y2)/2
            roi_w_half = roi_ctr_x - roi[1]
            roi_h_half = roi_ctr_y - roi[2]
            roi_w_half_new = self._global_context_scale * roi_w_half
            roi_h_half_new = self._global_context_scale * roi_h_half

            y[idx][0] = x[idx][0]  # cls
            y[idx][1] = roi_ctr_x - roi_w_half_new  # x1
            y[idx][2] = roi_ctr_y - roi_h_half_new  # y1
            y[idx][3] = roi_ctr_x + roi_w_half_new  # x2
            y[idx][4] = roi_ctr_y + roi_h_half_new  # y2
            y[idx] = self.clip_boxes(y[idx], im_info)
            if DEBUG:
                print('y[{}]:{}'.format(idx, y[idx].asnumpy()))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)


    @staticmethod
    def clip_boxes(box, im_shape):
        """
        Clip boxes to image boundaries.
        :param boxes: [1, 5]
        :param im_shape: tuple of 2
        :return: [1, 5]
        """
        # x1 >= 0
        box[1] = mx.nd.maximum(mx.nd.minimum(box[1], im_shape[0][1] - 1), 0)
        # y1 >= 0
        box[2] = mx.nd.maximum(mx.nd.minimum(box[2], im_shape[0][0] - 1), 0)
        # x2 < im_shape[1]
        box[3] = mx.nd.maximum(mx.nd.minimum(box[3], im_shape[0][1] - 1), 0)
        # y2 < im_shape[0]
        box[4] = mx.nd.maximum(mx.nd.minimum(box[4], im_shape[0][0] - 1), 0)
        return box


@mx.operator.register('roi_global_context')
class ROIGlobalContextProp(mx.operator.CustomOpProp):
    def __init__(self, global_context_scale = '1.2'):
        super(ROIGlobalContextProp, self).__init__(need_top_grad=False)
        self._global_context_scale =float(global_context_scale)

    def list_arguments(self):
        return ['rois','im_info']

    def list_outputs(self):
        return ['rois_output']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        im_info_shape = in_shape[1]
        output_rois_shape = rpn_rois_shape

        return [rpn_rois_shape, im_info_shape], \
               [output_rois_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ROIGlobalContextOperator(self._global_context_scale)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []