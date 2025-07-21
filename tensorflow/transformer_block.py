import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dropout
from multi_head_self_attention import MultiHeadSelfAttention
from feed_forward_network import FeedForwardNetwork

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, activation="relu", **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)  # Pass extra kwargs to the parent class constructor
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(ff_dim, embed_dim, activation=activation)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    @classmethod
    def from_config(cls, config):
        # Handle any unexpected arguments during deserialization
        config = {key: value for key, value in config.items() if key not in {"name", "trainable", "dtype"}}  # Exclude unwanted keys
        return cls(**config)

    def get_config(self):
        # Return only the essential parameters for serialization
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.att.embed_dim,
            'num_heads': self.att.num_heads,
            'ff_dim': self.ffn.ff_dim,
            'rate': self.dropout1.rate,
            'activation': self.ffn.activation
        })
        return config
