##
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils

from tensorflow.python.ops import image_ops

class RandomContrast(Layer):
  """Adjust the contrast of an image or images by a random factor.
  Contrast is adjusted independently for each channel of each image during
  training.
  For each channel, this layer computes the mean of the image pixels in the
  channel and then adjusts each component `x` of each pixel to
  `(x - mean) * contrast_factor + mean`.
  Input shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.
  Output shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.
  Attributes:
    factor: a positive float represented as fraction of value, or a tuple of
      size 2 representing lower and upper bound. When represented as a single
      float, lower = upper. The contrast factor will be randomly picked between
      [1.0 - lower, 1.0 + upper].
    seed: Integer. Used to create a random seed.
    name: A string, the name of the layer.
  Raise:
    ValueError: if lower bound is not between [0, 1], or upper bound is
      negative.
  """

  def __init__(self, factor, seed=None, name=None, **kwargs):
    self.factor = factor
    if isinstance(factor, (tuple, list)):
      self.lower = factor[0]
      self.upper = factor[1]
    else:
      self.lower = self.upper = factor
    if self.lower < 0. or self.upper < 0. or self.lower > 1.:
      raise ValueError('Factor cannot have negative values, '
                       'got {}'.format(factor))
    self.seed = seed
    self.input_spec = InputSpec(ndim=4)
    super(RandomContrast, self).__init__(name=name, **kwargs)

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    def random_contrasted_inputs():
      return image_ops.random_contrast(inputs, 1. - self.lower, 1. + self.upper,
                                       self.seed)

    output = tf_utils.smart_cond(training, random_contrasted_inputs,
                                 lambda: inputs)
    output.set_shape(inputs.shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'factor': self.factor,
        'seed': self.seed,
    }
    base_config = super(RandomContrast, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))