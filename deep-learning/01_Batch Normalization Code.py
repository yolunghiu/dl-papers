import numpy as np


def batchnorm_forward(x, gamma, beta, bn_param):
    """针对全连接层的
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']		# train or test
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape		# N个样本，该层有D个神经元
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
    	# 训练阶段： 1. 把输入的x标准化 2. 更新全局的均值方差 3. 将bp时用到的变量缓存起来
        mean = np.mean(x,axis = 0)
        var = np.var(x,axis = 0)
        x_norm = (x-mean)/np.sqrt(var + eps)
        out = x_norm * gamma + beta

        running_mean = running_mean * momentum + (1-momentum) * mean
        running_var = running_var * momentum + (1-momentum) * var

        cache = (x_norm,x,mean,var,beta,gamma,eps)
    elif mode == 'test':
        x_norm = (x-running_mean)/np.sqrt(running_var + eps)
        out = x_norm* gamma + beta

        cache = (out,x,running_mean,running_var,beta,gamma,eps)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """这是按论文上的公式实现的，那篇博客里是根据计算图反向传播一步一步计算的，论文中的公式更快一些
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x_normalized, x, sample_mean, sample_var, beta, gamma, eps = cache
    N, D = x.shape
    dx_normalized = dout * gamma       # [N,D]
    x_mu = x - sample_mean             # [N,D]
    sample_std_inv = 1.0 / np.sqrt(sample_var + eps)    # [1,D]
    dsample_var = -0.5 * np.sum(dx_normalized * x_mu, axis=0, keepdims=True) * sample_std_inv**3
    dsample_mean = -1.0 * np.sum(dx_normalized * sample_std_inv, axis=0, keepdims=True) - \
                   2.0 * dsample_var * np.mean(x_mu, axis=0, keepdims=True)
    dx1 = dx_normalized * sample_std_inv
    dx2 = 2.0/N * dsample_var * x_mu
    dx = dx1 + dx2 + 1.0/N * dsample_mean
    dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)

    return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """针对卷积层的

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape
    x_new = x.transpose(0, 2, 3, 1).reshape(N*H*W, C)#分成不同的channel来算，所以可以直接用之前的代码
    out, cache = batchnorm_forward(x_new, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
 	N, C, H, W = dout.shape
    dout_new = dout.transpose(0, 2, 3, 1).reshape(N*H*W, C)
    dx, dgamma, dbeta = batchnorm_backward(dout_new, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return dx, dgamma, dbeta


