import tensorflow as tf
from . import config
from .. import ops


class TransformerLayer(tf.keras.models.Model):
    """
    Single Transformer block implementing MHA
    """
    def __init__(self, ff_units, d_model, num_heads, dropout, causal=False, conv_filter=1, conv_padding="same",
                 efficient_attention=False, shared_qk=False, activation="relu", fused_qkv=False, name="TransformerLayer"):
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
        self.conv_filter = conv_filter
        self.conv_padding = conv_padding
        self.fused_qkv = fused_qkv
        if self.fused_qkv:
            self.mha = MultiHeadAttention(self.d_model, self.num_heads,
                                          causal=self.causal, efficient=self.efficient_attention, shared_qk=self.shared_qk,
                                          name=self.name+"_MultiHeadAttention")
        else:
            self.mha = MultiHeadAttentionFusedQKV(self.d_model, self.num_heads,
                                                  causal=self.causal, efficient=self.efficient_attention,
                                                  name=self.name+"_MultiHeadAttention")
        self.dropout_1 = tf.keras.layers.Dropout(rate=self.dropout)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1 = tf.keras.layers.Conv1D(filters=self.ff_units, kernel_size=self.conv_filter, padding=self.conv_padding)
        self.act = tf.keras.layers.Activation(self.activation)
        self.ffn_2 = tf.keras.layers.Conv1D(filters=self.d_model, kernel_size=self.conv_filter, padding=self.conv_padding)
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
        
        outputs = self.ffn_2(self.act(self.ffn_1(attention)))
        outputs = self.dropout_2(outputs)
        
        outputs = self.layer_norm_2(outputs)
            
        return outputs
    
    def get_config(self):
        return {"ff_units": self.ff_units,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "causal": self.causal,
                "efficient_attention": self.efficient_attention,
                "shared_qk": self.shared_qk,
                "conv_padding": self.conv_padding,
                "conv_filter": self.conv_filter,
                "fused_qkv": self.fused_qkv,
                "activation": self.activation}
    
    
class RevTransformerLayer(tf.keras.Model):
    def __init__(self, ff_units, d_model, num_heads, dropout, name="RevTransformerLayer",
                 causal=False, conv_filter=1, conv_padding="same",
                 efficient_attention=False, shared_qk=False, activation="relu"):
        super(RevTransformerLayer, self).__init__(name=name)
        assert d_model%2 == 0
        assert num_heads%2 == 0
        self.ff_units = ff_units
        self.d_model = int(d_model/2)
        self.num_heads = int(num_heads/2)
        self.dropout = dropout
        self.causal = causal
        self.conv_filter = conv_filter
        self.conv_padding = conv_padding
        self.efficient_attention = efficient_attention
        self.shared_qk = shared_qk
        self.activation = activation
        self.f = TransformerLayer(ff_units=self.ff_units,
                                  d_model=self.d_model,
                                  num_heads=self.num_heads,
                                  dropout=self.dropout,
                                  causal=self.causal,
                                  conv_filter=self.conv_filter,
                                  conv_padding=self.conv_padding,
                                  activation=self.activation,
                                  efficient_attention=self.efficient_attention,
                                  shared_qk=self.shared_qk,
                                  name=self.name+"_F")
        self.g = TransformerLayer(ff_units=self.ff_units,
                                  d_model=self.d_model,
                                  num_heads=self.num_heads,
                                  dropout=self.dropout,
                                  causal=self.causal,
                                  conv_filter=self.conv_filter,
                                  conv_padding=self.conv_padding,
                                  activation=self.activation,
                                  efficient_attention=self.efficient_attention,
                                  shared_qk=self.shared_qk,
                                  name=self.name+"_G")
        
    def call(self, inputs):
        token_inputs, mask = inputs["token_inputs"], inputs["mask_inputs"]
        token_inputs_1, token_inputs_2 = tf.split(token_inputs, num_or_size_splits=2, axis=-1)
        
        f_x2 = self.f(({"token_inputs": token_inputs_2,
                        "mask_inputs": mask}))
        y1 = f_x2 + token_inputs_1
        g_y1 = self.g(({"token_inputs": y1,
                        "mask_inputs": mask}))
        y2 = g_y1 + token_inputs_2
        return tf.concat([y1, y2], axis=-1)
    
    def backward_grads(self, y, dy, training=True):
        dy1, dy2 = dy
        y1, y2 = y

        with tf.GradientTape() as gtape:
            gtape.watch(y1)
            gy1 = self.g(y1, training=training)
        grads_combined = gtape.gradient(gy1, [y1] + self.g.trainable_variables,
                                        output_gradients=dy2)
        dg = grads_combined[1:]
        dx1 = dy1 + grads_combined[0]
        x2 = y2 - gy1

        with tf.GradientTape() as ftape:
            ftape.watch(x2)
            fx2 = self.f(x2, training=training)
        grads_combined = ftape.gradient(fx2, [x2] + self.f.trainable_variables,
                                        output_gradients=dx1)
        df = grads_combined[1:]
        dx2 = dy2 + grads_combined[0]
        x1 = y1 - fx2

        x = x1, x2
        dx = dx1, dx2
        grads = df + dg

        return x, dx, grads
    
    def get_config(self):
        return {"ff_units": self.ff_units,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "causal": self.causal,
                "efficient_attention": self.efficient_attention,
                "shared_qk": self.shared_qk,
                "conv_padding": self.conv_padding,
                "conv_filter": self.conv_filter,
                "activation": self.activation}

    
class TransformerStack(tf.keras.models.Model):
    """
    Build a stack of Transformer layers as Encoder/Decoder
    """
    def __init__(self, layers, ff_units, d_model, num_heads, dropout, causal=False, conv_filter=1, conv_padding="same",
                 efficient_attention=False, shared_qk=False, activation="relu", weight_sharing=False, reversible=False,
                 fused_qkv=False, name="TransformerStack"):
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
        self.conv_filter = conv_filter
        self.conv_padding = conv_padding
        self.reversible = reversible
        self.fused_qkv = fused_qkv
        
        if self.reversible:
            TLayer = RevTransformerLayer
        else:
            TLayer = TransformerLayer
        
        if self.weight_sharing:
            # create one layer and re-use
            self.xfmer_layer = TLayer(ff_units=self.ff_units,
                                      d_model=self.d_model,
                                      num_heads=self.num_heads,
                                      dropout=self.dropout,
                                      causal=self.causal,
                                      efficient_attention=self.efficient_attention,
                                      shared_qk=self.shared_qk,
                                      activation=self.activation,
                                      conv_filter=self.conv_filter,
                                      conv_padding=self.conv_padding,
                                      fused_qkv=self.fused_qkv,
                                      name=self.name+"_Layer")
            self.xfmer_layers = [self.xfmer_layer for i in range(self.tlayers)]
        else:
            # create a list of layers
            self.xfmer_layers = [
                TLayer(ff_units=self.ff_units,
                       d_model=self.d_model,
                       num_heads=self.num_heads,
                       dropout=self.dropout,
                       causal=self.causal,
                       efficient_attention=self.efficient_attention,
                       shared_qk=self.shared_qk,
                       activation=self.activation,
                       conv_filter=self.conv_filter,
                       conv_padding=self.conv_padding,
                       fused_qkv=self.fused_qkv,
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
                "efficient_attention": self.efficient_attention,
                "shared_qk": self.shared_qk,
                "fused_qkv": self.fused_qkv,
                "weight_sharing": self.weight_sharing}
    
    
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
    
    