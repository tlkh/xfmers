import tensorflow as tf
from . import layers


def Transformer(transformer_config):
    inputs = tf.keras.Input(shape=(transformer_config.max_seq_len, ), name="inputs")
    padding_mask = layers.PaddingMaskGenerator()(inputs)
    embeddings = layers.TokenPosEmbedding(transformer_config)(inputs)
    decoder_block = layers.TransformerStack(transformer_config,
                                            name="DecoderBlock")
    dec_outputs = decoder_block({"token_inputs": embeddings,
                                 "mask_inputs": padding_mask})
    preds = tf.keras.layers.Dense(transformer_config.vocab_size, name="outputs")(dec_outputs)
    return tf.keras.Model(inputs=inputs, outputs=preds, name=transformer_config.name)
