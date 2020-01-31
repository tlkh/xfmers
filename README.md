# xfmers

Quickly initialize bespoke Transformer models

## About

The goal of the xfmers library is to provide a simple API to quickly initialise Transformers with specified hyperparameters and features. The library generates standard TF 2.0 Keras models that can be used with `model.fit()`, Automatic Mixed Precision (AMP), XLA, Horovod and tf.distribute APIs.

**Included layers/features:**

* Multi-head attention
  * Toggle for Encoder or Decoder (causal) mode
* Transformer layers
* Transformer Stack (Encoder/Decoder)
  * Toggle for weight sharing (ALBERT-like)
* Embedding Layer
  * Learnable positional embeddings
  * Factorized embedding parameterization (ALBERT-like)

## Usage

**Creating an ALBERT-like Transformer**

```python
inputs = tf.keras.Input(shape=(None, ), name="inputs")
padding_mask = layers.PaddingMaskGenerator()(inputs)
embeddings = layers.TokenPosEmbedding(d_vocab=vocab_size, d_model=128, pos_length=512,
                                      # project embedding from 128 -> 512
                                      d_projection=512)(inputs)

# build encoder
encoder_block = layers.TransformerStack(layers=3,
                                        ff_units=2048,
                                        d_model=512,
                                        num_heads=8,
                                        dropout=0.1,
                                        causal=False,        # attend pair-wise between all positons
                                        activation=ops.gelu,
                                        weight_sharing=True, # share weights between all encoder layers
                                        name="EncoderBlock")
enc_outputs = encoder_block({"token_inputs": embeddings,
                             "mask_inputs": padding_mask})

# build decoder (causal)
decoder_block = layers.TransformerStack(layers=3,
                                        ff_units=2048,
                                        d_model=512,
                                        num_heads=8,
                                        dropout=0.1,
                                        causal=True,         # cannot attend to "future" positions
                                        activation=ops.gelu,
                                        weight_sharing=True, # share weights between all decoder layers
                                        name="DecoderBlock")
dec_outputs = decoder_block({"token_inputs": enc_outputs,
                             "mask_inputs": padding_mask})

# language modelling head
preds = layers.LMHead(vocab_size=vocab_size, name="outputs")(dec_outputs)

# Keras model
model = tf.keras.Model(inputs=inputs, outputs=preds, name=name)
```

## Installing Xfmers

**Install from Pip**

```shell
# coming soon
pip install xfmers
```

**Install from source**

```shell
# coming soon
```

## Support

* Core Maintainer: [Timothy Liu (tlkh)](https://github.com/tlkh)
* Please open an issue if you encounter problems or have a feature request

