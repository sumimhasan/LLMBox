import tensorflow as tf
from tensorflow.keras.layers import Dense

class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, ff_dim, embed_dim, activation="relu"):
        super(FeedForwardNetwork, self).__init__()
        self.ff_dim = ff_dim
        self.embed_dim = embed_dim
        self.activation = activation  # Store the activation function
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation=activation),
            Dense(embed_dim)  
        ])

    def call(self, inputs):
        return self.ffn(inputs)

    def get_config(self):
        config = super(FeedForwardNetwork, self).get_config()
        config.update({
            'ff_dim': self.ff_dim,            # Include feedforward dimension
            'embed_dim': self.embed_dim,      # Include embedding dimension
            'activation': self.activation     # Include activation function
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            ff_dim=config['ff_dim'],
            embed_dim=config['embed_dim'],
            activation=config['activation']
        )
