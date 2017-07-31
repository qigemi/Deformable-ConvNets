import mxnet as mx

bn_momentum = 0.922

def BK(data):
    return mx.symbol.BlockGrad(data=data)

# - - - - - - - - - - - - - - - - - - - - - - -
# Fundamental Elements
def BN(data, fix_gamma=False, momentum=bn_momentum, name=None,use_global_stats=True):
    bn     = mx.symbol.BatchNorm( data=data, fix_gamma=fix_gamma, momentum=bn_momentum,use_global_stats = use_global_stats, name=('%s__bn'%name),eps = 0.001)
    return bn

def AC(data, act_type='relu', name=None):
    act    = mx.symbol.Activation(data=data, act_type=act_type, name=('%s__%s' % (name, act_type)))
    return act

def BN_AC(data, momentum=bn_momentum, name=None):
    bn     = BN(data=data, name=name, fix_gamma=False, momentum=momentum)
    bn_ac  = AC(data=bn,   name=name)
    return bn_ac

def Conv(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, no_bias=True, w=None, b=None, attr=None, num_group=1,dilate = (1,1)):
    Convolution = mx.symbol.Convolution
    if w is None:
        conv     = Convolution(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=('%s__conv' %name), no_bias=no_bias, attr=attr,dilate = dilate)
    else:
        if b is None:
            conv = Convolution(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=('%s__conv' %name), no_bias=no_bias, weight=w, attr=attr,dilate = dilate)
        else:
            conv = Convolution(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=('%s__conv' %name), no_bias=False, bias=b, weight=w, attr=attr,dilate = dilate)
    return conv

# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Common functions < CVPR >
def Conv_BN(   data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    cov    = Conv(   data=data,   num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    cov_bn = BN(     data=cov,    name=('%s__bn' % name))
    return cov_bn

def Conv_BN_AC(data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    cov_bn = Conv_BN(data=data,   num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    cov_ba = AC(     data=cov_bn, name=('%s__ac' % name))
    return cov_ba

# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Common functions < ECCV >
def BN_Conv(   data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    bn     = BN(     data=data,   name=('%s__bn' % name))
    bn_cov = Conv(   data=bn,     num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    return bn_cov

def AC_Conv(   data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1,dilate = (1,1)):
    ac     = AC(     data=data,   name=('%s__ac' % name))
    ac_cov = Conv(   data=ac,     num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    return ac_cov

def BN_AC_Conv(data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1,dilate = (1,1)):
    bn     = BN(     data=data,   name=('%s__bn' % name))
    ba_cov = AC_Conv(data=bn,     num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr,dilate = (1,1))
    return ba_cov
