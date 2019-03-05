import tensorflow as tf

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('NHWC' or 'NCHW').
      mode: The mode for tf.pad.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if kwargs['data_format'] == 'NCHW':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end]],
                               mode=mode)
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs

def _darknet53_block(inputs, filters):
    shortcut = inputs
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)

    inputs = inputs + shortcut
    return inputs

def darknet53(inputs):
    """
    Builds Darknet-53 model.
    """
    inputs = _conv2d_fixed_padding(inputs, 32, 3)
    inputs = _conv2d_fixed_padding(inputs, 64, 3, strides=2)
    inputs = _darknet53_block(inputs, 32)
    inputs = _conv2d_fixed_padding(inputs, 128, 3, strides=2)

    for i in range(2):
        inputs = _darknet53_block(inputs, 64)

    inputs = _conv2d_fixed_padding(inputs, 256, 3, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 128)

    route_1 = inputs
    inputs = _conv2d_fixed_padding(inputs, 512, 3, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 256)

    route_2 = inputs
    inputs = _conv2d_fixed_padding(inputs, 1024, 3, strides=2)

    for i in range(4):
        inputs = _darknet53_block(inputs, 512)

    return route_1, route_2, inputs

def yolo_v3(inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
    """
    Creates YOLO v3 model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined. The channel order is RGB.
    :param num_classes: number of predicted classes.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :return:
    """
    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]

    # transpose the inputs(Num_samples x Height x Width x Channels) to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    # set batch norm params
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], data_format=data_format, reuse=reuse):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            biases_initializer=None,
                            activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):
            with tf.variable_scope('darknet-53'):
                route_1, route_2, inputs = darknet53(inputs)

            with tf.variable_scope('yolo-v3'):
                route, inputs = _yolo_block(inputs, 512)
                detect_1 = _detection_layer(
                    inputs, num_classes, _ANCHORS[6:9], img_size, data_format)
                detect_1 = tf.identity(detect_1, name='detect_1')

                inputs = _conv2d_fixed_padding(route, 256, 1)
                upsample_size = route_2.get_shape().as_list()
                inputs = _upsample(inputs, upsample_size, data_format)
                inputs = tf.concat([inputs, route_2],
                                   axis=1 if data_format == 'NCHW' else 3)

                route, inputs = _yolo_block(inputs, 256)

                detect_2 = _detection_layer(
                    inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
                detect_2 = tf.identity(detect_2, name='detect_2')

                inputs = _conv2d_fixed_padding(route, 128, 1)
                upsample_size = route_1.get_shape().as_list()
                inputs = _upsample(inputs, upsample_size, data_format)
                inputs = tf.concat([inputs, route_1],
                                   axis=1 if data_format == 'NCHW' else 3)

                _, inputs = _yolo_block(inputs, 128)

                detect_3 = _detection_layer(
                    inputs, num_classes, _ANCHORS[0:3], img_size, data_format)
                detect_3 = tf.identity(detect_3, name='detect_3')

                detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
                #when it is necessary to assign a name to ops that do not have a name
                detections = tf.identity(detections, name='detections')
                return detections