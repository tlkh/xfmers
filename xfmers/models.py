import tensorflow.compat.v2 as tf
from . import ops
from . import layers


def Transformer(vocab_size, dec_layers, ff_units, d_model, num_heads, dropout, max_seq_len=512, causal=False,
                       weight_sharing=False, efficient_attention=False, shared_qk=False, activation=ops.gelu,
                       conv_filter=1, conv_padding="same", reversible=False, fused_qkv=False, name="Transformer"):
    inputs = tf.keras.Input(shape=(None, ), name="inputs")
    padding_mask = layers.PaddingMaskGenerator()(inputs)
    embeddings = layers.TokenPosEmbedding(d_vocab=vocab_size, d_model=d_model, pos_length=max_seq_len, scale=1)(inputs)
    
    decoder_block = layers.TransformerStack(layers=dec_layers,
                                            ff_units=ff_units,
                                            d_model=d_model,
                                            num_heads=num_heads,
                                            dropout=dropout,
                                            causal=causal,
                                            activation=activation,
                                            weight_sharing=weight_sharing,
                                            conv_filter=conv_filter,
                                            conv_padding=conv_padding,
                                            reversible=reversible,
                                            fused_qkv=fused_qkv,
                                            name="DecoderBlock")
    dec_outputs = decoder_block({"token_inputs": embeddings,
                                 "mask_inputs": padding_mask})
    
    l_dropout = tf.keras.layers.Dropout(rate=dropout)(dec_outputs)
    
    preds = tf.keras.layers.Dense(vocab_size, name="outputs")(l_dropout)
    
    return tf.keras.Model(inputs=inputs, outputs=preds, name=name)
