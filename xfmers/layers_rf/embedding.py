import tensorflow as tf
from . import config
from .. import ops


class TokenPosEmbedding(tf.keras.layers.Layer):
    """
    Transformer Token and Position Embedding layer
    """
    def __init__(self, d_vocab, d_model, pos_length=512, scale=1, d_projection=None, name="TokenPosEmbedding"):
        """
        Initialize Transformer Embedding layer for token and position embeddings.
        This must be the very first layer in any model after the Input layer.
        Args:
            d_vocab: (int) Input vocabulary size
            d_model: (int) Dimension of embeddings
            pos_length: (int) Maximum length of position embeddings
            scale: (int) Apply scaling of int to token embeddings. Defaults to 1
            d_projection: (int) Factorized embedding parameterization (ALBERT-like). Projects output to another dimension
        Inputs:
            token_inputs: 1D Tensor of integer numbers (< d_vocab) and of maximum length pos_length
        Outputs:
            N-D tensor with shape: `(batch_size, ..., d_model)`
        """
        super(TokenPosEmbedding, self).__init__(name=name)
        self.d_vocab = d_vocab
        self.d_model = d_model
        self.pos_length = pos_length
        self.scale = scale
        self.d_projection = d_projection
        
        self.token_embeddings = tf.keras.layers.Embedding(self.d_vocab, self.d_model)
        self.pos_embeddings = tf.keras.layers.Embedding(self.pos_length, self.d_model)
        if self.d_projection:
            self.projection = tf.keras.layers.Dense(units=self.d_projection)

    def call(self, inputs):
        pos_range = ops.position_range(tf.shape(inputs)[-1])

        self.token_vector = self.token_embeddings(inputs)
        self.token_vector *= self.scale
        self.pos_vector = self.pos_embeddings(pos_range)

        embeddings = self.token_vector + self.pos_vector
        
        if self.d_projection:
            embeddings = self.projection(embeddings)
        
        return embeddings
    
    def get_config(self):
        return {"d_vocab": self.d_vocab,
                "d_model": self.d_model,
                "pos_length": self.pos_length,
                "d_projection": self.d_projection,
                "scale": self.scale}