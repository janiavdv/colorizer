from tensorflow import expand_dims, range, float32, math, reduce_sum, repeat, reshape, exp, nn

"""
Apply gaussian filter to a symbolic tensor.

src: https://stackoverflow.com/a/65219530
"""

def get_gaussian_kernel(shape=(3,3), sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    x = expand_dims(range(-n,n+1,dtype=float32),1)
    y = expand_dims(range(-m,m+1,dtype=float32),0)
    h = exp(math.divide_no_nan(-((x*x) + (y*y)), 2*sigma*sigma))
    h = math.divide_no_nan(h,reduce_sum(h))
    return h

def gaussian_blur(inp, shape=(3,3), sigma=0.5):
    in_channel = shape(inp)[-1]
    k = get_gaussian_kernel(shape,sigma)
    k = expand_dims(k,axis=-1)
    k = repeat(k,in_channel,axis=-1)
    k = reshape(k, (*shape, in_channel, 1))
    conv = nn.depthwise_conv2d(inp, k, strides=[1,1,1,1],padding="SAME")
    return conv