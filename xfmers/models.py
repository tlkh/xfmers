import tensorflow.compat.v2 as tf
from . import ops
from . import layers


def DecoderTransformer(vocab_size, dec_layers, ff_units, d_model, num_heads, dropout, max_seq_len=512, weight_sharing=False, name="DecoderTransformer"):
    inputs = tf.keras.Input(shape=(None, ), name="inputs")
    padding_mask = layers.PaddingMaskGenerator()(inputs)
    embeddings = layers.TokenPosEmbedding(d_vocab=vocab_size, d_model=d_model, pos_length=max_seq_len, scale=d_model**0.5)(inputs)
    
    decoder_block = layers.TransformerStack(layers=dec_layers,
                                            ff_units=ff_units,
                                            d_model=d_model,
                                            num_heads=num_heads,
                                            dropout=dropout,
                                            causal=True,
                                            activation=ops.gelu,
                                            weight_sharing=weight_sharing,
                                            name="DecoderBlock")
    dec_outputs = decoder_block({"token_inputs": embeddings,
                                 "mask_inputs": padding_mask})
    
    preds = layers.LMHead(vocab_size=vocab_size, name="outputs")(dec_outputs)
    
    return tf.keras.Model(inputs=inputs, outputs=preds, name=name)
