import numpy as np
import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.embed_dim = embed_dim

        # Compute positional encodings
        pos = np.arange(position)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pe = np.zeros((position, embed_dim))
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)
        self.positional_encoding = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]

    @classmethod
    def from_config(cls, config):
        # Ensure that only the essential parameters are passed to the constructor
        return cls(config['position'], config['embed_dim'])

    def get_config(self):
        # Only include the essential parameters (position and embed_dim) in the config
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'position': self.position,
            'embed_dim': self.embed_dim
        })
        return config
