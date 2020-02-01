<div align="center">
<h1>xfmers</h1>
    <p>Quickly initialize bespoke Transformer models</p>
    <code>pip install xfmers</code>
</div>

## About

The goal of the xfmers library is to provide a simple API to quickly initialise Transformers with specified hyperparameters and features. The library generates standard TF 2.0 Keras models that can be used with `model.fit()`, Automatic Mixed Precision (AMP), XLA, Horovod and tf.distribute APIs etc.

**Included layers/features:**

* Multi-head attention
  * Encoder or Decoder (causal) mode
  * [Efficient attention mode](https://arxiv.org/abs/1812.01243)
* Transformer layers
  * Choice to use spatial convolution
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

Models can be created using Keras layers and trained using `model.fit()` or Gradient Tape.

**Creating an ALBERT-like Transformer**

```python
inputs = tf.keras.Input(shape=(None, ), name="inputs")
padding_mask = layers.PaddingMaskGenerator()(inputs)
embeddings = layers.TokenPosEmbedding(d_vocab=vocab_size, d_model=128, pos_length=max_seq_len, scale=d_model**0.5,
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
                                        conv_filter=1,
                                        conv_padding="same",
                                        reversible=False,
                                        name="EncoderBlock")
enc_outputs = encoder_block({"token_inputs": embeddings,
                             "mask_inputs": padding_mask})
preds = layers.LMHead(vocab_size=vocab_size, name="outputs")(enc_outputs)

model = tf.keras.Model(inputs=inputs, outputs=preds, name=name)
```

To convert this into a GPT-like Transformer, one would only need to set `d_projection=None,causal=True,weight_sharing=False`.

## Support

* Core Maintainer: [Timothy Liu (tlkh)](https://github.com/tlkh)
* Please open an issue if you encounter problems or have a feature request

