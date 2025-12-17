import qkeras
import tensorflow as tf

## WARNING: This code doesn't actually get used. It needs to be placed in qkeras/quantizers.py so that it can be properly registered by qkeras.
class see_quantizer(quantized_bits):

    def __init__(self, flip_tensor=None, random_flip_rate=0.0, **kwargs):
      super(see_quantizer, self).__init__(**kwargs)
      self.flip_tensor = flip_tensor
      self.random_flip_rate = random_flip_rate
  
    def __call__(self, inputs):
      x_qua = super(see_quantizer, self).__call__(inputs)
      if self.integer != 0 or isinstance(self.alpha, six.string_types) or self.symmetric or not self.keep_negative:
        raise NotImplementedError("see_quantizer only supports simple cases for now.")
      integer_representation = tf.cast(tf.round(x_qua * (2 ** (self.bits - 1))), tf.int32)
      flip_tensor = self.flip_tensor
      if not flip_tensor:
        flip_tensor = tf.zeros_like(integer_representation)
      if self.random_flip_rate > 0.0:
        random_flip_map = tf.cast(tf.random.uniform(tf.concat([tf.shape(integer_representation), [self.bits]], axis=0), minval=0, maxval=1) < self.random_flip_rate, tf.int32)
        random_tensor = tf.reduce_sum(random_flip_map * tf.cast(2 ** tf.range(self.bits), tf.int32), axis=-1)
        flip_tensor = flip_tensor ^ random_tensor
      integer_representation = integer_representation ^ flip_tensor
      return tf.cast(integer_representation, tf.float32) / (2 ** (self.bits - 1))