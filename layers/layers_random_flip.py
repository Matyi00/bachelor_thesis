##
import tensorflow as tf
from tensorflow.python.keras import backend as K

from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils

from tensorflow.python.ops import image_ops

from tensorflow.python.ops import stateful_random_ops


HORIZONTAL = 'horizontal'
VERTICAL = 'vertical'
HORIZONTAL_AND_VERTICAL = 'horizontal_and_vertical'


class RandomFlip(tf.keras.layers.Layer):
  """Randomly flip each image horizontally and vertically.
  This layer will flip the images based on the `mode` attribute.
  During inference time, the output will be identical to input. Call the layer
  with `training=True` to flip the input.
  Input shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.
  Output shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.
  Attributes:
    mode: String indicating which flip mode to use. Can be "horizontal",
      "vertical", or "horizontal_and_vertical". Defaults to
      "horizontal_and_vertical".
    seed: Integer. Used to create a random seed.
    name: A string, the name of the layer.
  """

  def __init__(self,
               mode=HORIZONTAL_AND_VERTICAL,
               seed=None,
               name=None,
               **kwargs):
    super(RandomFlip, self).__init__(name=name, **kwargs)
    self.mode = mode
    if mode == HORIZONTAL:
      self.horizontal = True
      self.vertical = False
    elif mode == VERTICAL:
      self.horizontal = False
      self.vertical = True
    elif mode == HORIZONTAL_AND_VERTICAL:
      self.horizontal = True
      self.vertical = True
    else:
      raise ValueError('RandomFlip layer {name} received an unknown mode '
                       'argument {arg}'.format(name=name, arg=mode))
    self.seed = seed
    self._rng = make_generator(self.seed)
    self.input_spec = InputSpec(ndim=4)

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()
    def random_flipped_inputs():
      flipped_outputs = inputs
      if self.horizontal:
        flipped_outputs = image_ops.random_flip_left_right(flipped_outputs,
                                                           self.seed)
      if self.vertical:
        flipped_outputs = image_ops.random_flip_up_down(
            flipped_outputs, self.seed)
      return flipped_outputs

    output = tf_utils.smart_cond(training, random_flipped_inputs,
                                 lambda: inputs)
    output.set_shape(inputs.shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'mode': self.mode,
        'seed': self.seed,
    }
    base_config = super(RandomFlip, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def make_generator(seed=None):
  if seed:
    return stateful_random_ops.Generator.from_seed(seed)
  else:
    return stateful_random_ops.Generator.from_non_deterministic_state()
