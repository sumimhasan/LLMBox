import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, Dense
from transformer_block import TransformerBlock
from positional_encoding import PositionalEncoding

# Function to add multiple Transformer Blocks
def add_transformer_layers(x, num_layers, embed_dim, num_heads, ff_dim, rate=0.1):
    for _ in range(num_layers):
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
        x = transformer_block(x, training=False)
    return x

# Main model building function
def build_transformer_model(num_words, maxlen, embed_dim, num_heads, ff_dim, num_layers=6, rate=0.1):
    inputs = tf.keras.Input(shape=(maxlen,))  # Define input layer

    # Embedding layer
    embedding_layer = Embedding(input_dim=num_words, output_dim=embed_dim, input_length=maxlen)
    embedded_sequences = embedding_layer(inputs)

    # Add Positional Encoding
    positional_encoding = PositionalEncoding(maxlen, embed_dim)
    x = positional_encoding(embedded_sequences)

    # Add multiple Transformer Blocks
    x = add_transformer_layers(x, num_layers, embed_dim, num_heads, ff_dim, rate)

    # Flatten and add Dense layers for output
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(num_words, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

