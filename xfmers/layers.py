import tensorflow.compat.v2 as tf
from . import ops


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head Attention (MHA) block
    """
    def __init__(self, d_model, num_heads, causal=False, efficient=False, shared_qk=False, name="MultiHeadAttention"):
        """
        Initialize MHA
        Args:
            d_model: (int) Dimension of hidden layers. Must be a multiple of `num_heads`
            num_heads: (int) Number of attention heads
            causal: (int) Disallow "looking into the future" by masking future time steps
            efficient: (bool) Alternative efficient implementation (https://arxiv.org/abs/1812.01243)
            shared_qk: (bool) Share Query and Key weights 
        Inputs:
            inputs: (dict) Dictionary with keys "query", "key", "value", "mask"
        Outputs:
            N-D tensor with shape: `(batch_size, ..., d_model)`
        """
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        self.causal = causal
        self.efficient = efficient
        self.shared_qk = shared_qk
        assert self.d_model % self.num_heads == 0
        self.depth = self.d_model // self.num_heads
        
        self.query_dense = tf.keras.layers.Dense(units=self.d_model)
        if shared_qk:
            self.key_dense = self.query_dense
        else:
            self.key_dense = tf.keras.layers.Dense(units=self.d_model)
        self.value_dense = tf.keras.layers.Dense(units=self.d_model)
        self.final_linear = tf.keras.layers.Dense(units=self.d_model)
        
        if self.efficient:
            self.attention_op = ops.efficient_attention
        else:
            self.attention_op = ops.scaled_dot_product_attention
        
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs,
                            shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs["query"], inputs["key"], inputs["value"], inputs["mask"]
        
        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        # shape: (batch, seq, d_model)

        # split heads
        batch_size = tf.shape(query)[0]
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        # shape: (batch, heads, seq, d_model//heads)

        if self.causal:
            mask_len = tf.shape(mask)[-1]
            causal_mask = ops.causal_attention_mask(mask_len)
            mask += causal_mask

        # scaled dot-product attention
        scaled_attention = self.attention_op(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))
        # final linear layer
        outputs = self.final_linear(concat_attention)
        return outputs
    
    def get_config(self):
        return {"d_model": self.d_model,
                "num_heads": self.num_heads,
                "efficient": self.efficient,
                "causal": self.causal}


class TransformerLayer(tf.keras.models.Model):
    """
    Single Transformer block implementing MHA
    """
    def __init__(self, ff_units, d_model, num_heads, dropout, causal=False, efficient_attention=False, shared_qk=False, activation="relu", name="TransformerLayer"):
        """
        Initialize single Transformer layer
        Args:
            d_model: (int) Dimension of hidden layers. Must be a multiple of `num_heads`
            ff_units: (int) Dimension of hidden layer in positional feed-forward unit
            num_heads: (int) Number of attention heads
            dropout: (float) dropout rate after MHA and feed-forward net
            causal: (int) Disallow MHA "looking into the future" by masking future time steps
            layer_norm: (string) Configuration of layer normalization. Defaults to "single" (GPT-2-like) or "double" (GPT-like)
            activation: (string) Activation function to use inside feed-forward net. Defaults to "relu"
        Inputs:
            inputs: (dict) Dictionary with keys "token_inputs" and "mask_inputs"
        Outputs:
            N-D tensor with shape: `(batch_size, ..., d_model)`
        """
        super(TransformerLayer, self).__init__(name=name)
        self.ff_units = ff_units
        self.d_model = d_model
        self.dropout = dropout
        self.num_heads = num_heads
        self.causal = causal
        self.efficient_attention = efficient_attention
        self.shared_qk = shared_qk
        self.activation = activation
        self.mha = MultiHeadAttention(self.d_model, self.num_heads,
                                      causal=self.causal, efficient=self.efficient_attention, shared_qk=self.shared_qk,
                                      name=self.name+"_MultiHeadAttention")
        self.dropout_1 = tf.keras.layers.Dropout(rate=self.dropout)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1 = tf.keras.layers.Dense(units=self.ff_units)
        self.relu = tf.keras.layers.Activation(self.activation)
        self.ffn_2 = tf.keras.layers.Dense(units=self.d_model)
        self.dropout_2 = tf.keras.layers.Dropout(rate=self.dropout)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs):
        token_inputs, mask = inputs["token_inputs"], inputs["mask_inputs"]
            
        attention = self.mha({"query": token_inputs,
                              "key": token_inputs,
                              "value": token_inputs,
                              "mask": mask})
        attention = token_inputs + self.dropout_1(attention)
        
        attention = self.layer_norm_1(attention)
        
        outputs = self.ffn_2(self.relu(self.ffn_1(attention)))
        outputs = self.dropout_2(outputs)
        
        outputs = self.layer_norm_2(outputs)
            
        return outputs
    
    def get_config(self):
        return {"ff_units": self.ff_units,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "causal": self.causal,
                "activation": self.activation}

    
class TransformerStack(tf.keras.models.Model):
    """
    Build a stack of Transformer layers as Encoder/Decoder
    """
    def __init__(self, layers, ff_units, d_model, num_heads, dropout, causal=False, efficient_attention=False, shared_qk=False, activation="relu", weight_sharing=False, name="TransformerStack"):
        """
        Initialize Transformer layer stack
        Args:
            layers: (int) Number of transformer layers in the stack
            d_model: (int) Dimension of hidden layers. Must be a multiple of `num_heads`
            ff_units: (int) Dimension of hidden layer in positional feed-forward unit
            num_heads: (int) Number of attention heads
            dropout: (float) dropout rate after MHA and feed-forward net
            causal: (int) Disallow MHA "looking into the future" by masking future time steps
            layer_norm: (string) Configuration of layer normalization. Defaults to "single" (GPT-2-like) or "double" (GPT-like)
            activation: (string) Activation function to use inside feed-forward net. Defaults to "relu"
            weight_sharing: (bool) Share weights between all Transformer layers. Defaults to False
        Inputs:
            inputs: (dict) Dictionary with keys "token_inputs" and "mask_inputs"
        Outputs:
            N-D tensor with shape: `(batch_size, ..., d_model)`
        """
        super(TransformerStack, self).__init__(name=name)
        self.tlayers = layers
        self.ff_units = ff_units
        self.d_model = d_model
        self.dropout = dropout
        self.num_heads = num_heads
        self.causal = causal
        self.efficient_attention = efficient_attention
        self.shared_qk = shared_qk
        self.activation = activation
        self.weight_sharing = weight_sharing
        
        if self.weight_sharing:
            # create one layer and re-use
            self.xfmer_layer = TransformerLayer(ff_units=self.ff_units,
                                                d_model=self.d_model,
                                                num_heads=self.num_heads,
                                                dropout=self.dropout,
                                                causal=self.causal,
                                                efficient_attention=self.efficient_attention,
                                                shared_qk=self.shared_qk,
                                                activation=self.activation,
                                                name=self.name+"_Layer")
            self.xfmer_layers = [self.xfmer_layer for i in range(self.tlayers)]
        else:
            # create a list of layers
            self.xfmer_layers = [
                TransformerLayer(ff_units=self.ff_units,
                                 d_model=self.d_model,
                                 num_heads=self.num_heads,
                                 dropout=self.dropout,
                                 causal=self.causal,
                                 efficient_attention=self.efficient_attention,
                                 shared_qk=self.shared_qk,
                                 activation=self.activation,
                                 name=self.name+"_Layer_"+str(i+1)) for i in range(self.tlayers)
            ]
    
    def call(self, inputs):
        outputs, mask = inputs["token_inputs"], inputs["mask_inputs"]
        
        for layer in self.xfmer_layers:
            outputs = layer({"token_inputs": outputs,
                             "mask_inputs": mask})
            
        return outputs
    
    def get_config(self):
        return {"layers": self.layers,
                "ff_units": self.ff_units,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "causal": self.causal,
                "activation": self.activation,
                "weight_sharing": self.weight_sharing}
    

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


class PaddingMaskGenerator(tf.keras.layers.Layer):
    """
    Generate padding mask for sequence
    """
    def __init__(self, name="PaddingMaskGenerator"):
        """
        Creates a mask for any padding (`0`) characters in the input sequence
        Args:
            None
        Inputs:
            token_inputs: 1D Tensor of integer numbers
        Outputs:
            N-D tensor
        """
        super(PaddingMaskGenerator, self).__init__(name=name)
        
    def call(self, inputs):
        return ops.create_padding_mask(inputs)
    
    def get_config(self):
        return {}

    
class LMHead(tf.keras.layers.Layer):
    """
    Language Modelling Head
    """
    def __init__(self, vocab_size, name="LMHead"):
        """
        LMHead
        Args:
            None
        Inputs:
            token_inputs: 1D Tensor of integer numbers
        Outputs:
            N-D tensor
        """
        super(LMHead, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.dense_softmax = tf.keras.layers.Dense(units=self.vocab_size, activation="softmax")
        
    def call(self, inputs):
        return self.dense_softmax(inputs[:,-1])
    
    def get_config(self):
        return {"vocab_size": self.vocab_size}
    