/*!
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file deformable_convolution.cu
 * \brief
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai
*/

#include "./bilinear-inl.h"
#include <vector>

namespace mxnet {
  namespace op {

    template<>
    Operator* CreateOp<gpu>(BilinearParam param, int dtype,
      std::vector<TShape> *in_shape,
      std::vector<TShape> *out_shape,
      Context ctx) {
      Operator *op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new BilinearOp<gpu, DType>(param);
      })
        return op;
    }

  }  // namespace op
}  // namespace mxnet

