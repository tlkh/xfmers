<div align="center">
<h1>xfmers</h1>
  <p>Quickly initialize bespoke Transformer models</p>
</div>

## About

The goal of the xfmers library is to provide a simple API to quickly initialise Transformers with specified hyperparameters and features. The library generates standard TF 2.0 Keras models that can be used with `model.fit()`, Automatic Mixed Precision (AMP), XLA, Horovod and tf.distribute APIs etc.

**Included layers/features:**

* Multi-head attention
  * Encoder or Decoder (causal) mode
* Transformer layers
  * Spatial convolution
  * Reversible layers
* Transformer Stack (Encoder/Decoder)
  * Weight sharing (ALBERT-like)
* Embedding Layer
  * Learnable positional embeddings
  * Factorized embedding parameterization (ALBERT-like)
* Misc
  * Training schedules
  * Activation functions

## Usage

**Creating an ALBERT-like Transformer**

Models can be created using Keras layers and trained using `model.fit()` or Gradient Tape.

```python
inputs = tf.keras.Input(shape=(None, ), name="inputs")
padding_mask = layers.PaddingMaskGenerator()(inputs)
embeddings = layers.TokenPosEmbedding(d_vocab=vocab_size, d_model=128, pos_length=512,
                                      # project embedding from 128 -> 512
                                      d_projection=512)(inputs)
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
preds = layers.LMHead(vocab_size=vocab_size, name="outputs")(enc_outputs)

model = tf.keras.Model(inputs=inputs, outputs=preds, name=name)
```

## Installing Xfmers

**Install from Pip**

```shell
pip install xfmers
```

## Support

* Core Maintainer: [Timothy Liu (tlkh)](https://github.com/tlkh)
* Please open an issue if you encounter problems or have a feature request

