from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateful_random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.util.tf_export import keras_export

class RandomTranslation(Layer):
  """Randomly translate each image during training.
  Arguments:
    height_factor: a positive float represented as fraction of value, or a tuple
      of size 2 representing lower and upper bound for shifting vertically. When
      represented as a single float, this value is used for both the upper and
      lower bound. For instance, `height_factor=(0.2, 0.3)` results in an output
      height varying in the range `[original - 20%, original + 30%]`.
      `height_factor=0.2` results in an output height varying in the range
      `[original - 20%, original + 20%]`.
    width_factor: a positive float represented as fraction of value, or a tuple
      of size 2 representing lower and upper bound for shifting horizontally.
      When represented as a single float, this value is used for both the upper
      and lower bound.
    fill_mode: Points outside the boundaries of the input are filled according
      to the given mode (one of `{'constant', 'reflect', 'wrap'}`).
      - *reflect*: `(d c b a | a b c d | d c b a)`
        The input is extended by reflecting about the edge of the last pixel.
      - *constant*: `(k k k k | a b c d | k k k k)`
        The input is extended by filling all values beyond the edge with the
        same constant value k = 0.
      - *wrap*: `(a b c d | a b c d | a b c d)`
        The input is extended by wrapping around to the opposite edge.
    interpolation: Interpolation mode. Supported values: "nearest", "bilinear".
    seed: Integer. Used to create a random seed.
    name: A string, the name of the layer.
  Input shape:
    4D tensor with shape: `(samples, height, width, channels)`,
      data_format='channels_last'.
  Output shape:
    4D tensor with shape: `(samples, height, width, channels)`,
      data_format='channels_last'.
  Raise:
    ValueError: if lower bound is not between [0, 1], or upper bound is
      negative.
  """

  def __init__(self,
               height_factor,
               width_factor,
               fill_mode='constant',
               interpolation='bilinear',
               seed=None,
               name=None,
               **kwargs):
    self.height_factor = height_factor
    if isinstance(height_factor, (tuple, list)):
      self.height_lower = abs(height_factor[0])
      self.height_upper = height_factor[1]
    else:
      self.height_lower = self.height_upper = height_factor
    if self.height_upper < 0.:
      raise ValueError('`height_factor` cannot have negative values as upper '
                       'bound, got {}'.format(height_factor))
    if abs(self.height_lower) > 1. or abs(self.height_upper) > 1.:
      raise ValueError('`height_factor` must have values between [-1, 1], '
                       'got {}'.format(height_factor))

    self.width_factor = width_factor
    if isinstance(width_factor, (tuple, list)):
      self.width_lower = abs(width_factor[0])
      self.width_upper = width_factor[1]
    else:
      self.width_lower = self.width_upper = width_factor
    if self.width_upper < 0.:
      raise ValueError('`width_factor` cannot have negative values as upper '
                       'bound, got {}'.format(width_factor))
    if abs(self.width_lower) > 1. or abs(self.width_upper) > 1.:
      raise ValueError('`width_factor` must have values between [-1, 1], '
                       'got {}'.format(width_factor))

    if fill_mode not in {'reflect', 'wrap', 'constant'}:
      raise NotImplementedError(
          'Unknown `fill_mode` {}. Only "constant" is '
          'supported.'.format(fill_mode))
    if interpolation not in {'nearest', 'bilinear'}:
      raise NotImplementedError(
          'Unknown `interpolation` {}. Only `nearest` and '
          '`bilinear` are supported.'.format(interpolation))
    self.fill_mode = fill_mode
    self.interpolation = interpolation
    self.seed = seed
    self._rng = make_generator(self.seed)
    self.input_spec = InputSpec(ndim=4)
    super(RandomTranslation, self).__init__(name=name, **kwargs)

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    def random_translated_inputs():
      """Translated inputs with random ops."""
      inputs_shape = array_ops.shape(inputs)
      batch_size = inputs_shape[0]
      h_axis, w_axis = 1, 2
      img_hd = math_ops.cast(inputs_shape[h_axis], dtypes.float32)
      img_wd = math_ops.cast(inputs_shape[w_axis], dtypes.float32)
      height_translate = self._rng.uniform(
          shape=[batch_size, 1],
          minval=-self.height_lower,
          maxval=self.height_upper)
      height_translate = height_translate * img_hd
      width_translate = self._rng.uniform(
          shape=[batch_size, 1],
          minval=-self.width_lower,
          maxval=self.width_upper)
      width_translate = width_translate * img_wd
      translations = array_ops.concat([height_translate, width_translate], axis=1)
      return image_ops.scale_and_translate(inputs,(img_hd,img_wd),(1,1),(height_translate[0][0],width_translate[0][0]))
      # return transform(
      #     inputs,
      #     get_translation_matrix(translations),
      #     interpolation=self.interpolation,
      #     fill_mode=self.fill_mode)

    output = tf_utils.smart_cond(training, random_translated_inputs,
                                 lambda: inputs)
    output.set_shape(inputs.shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'height_factor': self.height_factor,
        'width_factor': self.width_factor,
        'fill_mode': self.fill_mode,
        'interpolation': self.interpolation,
        'seed': self.seed,
    }
    base_config = super(RandomTranslation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def make_generator(seed=None):
  if seed:
    return stateful_random_ops.Generator.from_seed(seed)
  else:
    return stateful_random_ops.Generator.from_non_deterministic_state()

