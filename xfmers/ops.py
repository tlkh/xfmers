import tensorflow.compat.v2 as tf


def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], matmul_qk.dtype)
    logits = matmul_qk / tf.math.sqrt(depth)
    # add the mask to zero out padding tokens
    mask = tf.cast(mask, logits.dtype)
    logits += (mask * -1e9)
    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    # sns.heatmap(attention_weights.numpy()[0,0,:,:])
    # plt.show()
    output = tf.matmul(attention_weights, value)
    return output


def efficient_attention(query, key, value, mask):
    """
    Efficient Attention: Attention with Linear Complexities
    https://arxiv.org/abs/1812.01243
    """
    matmul_vk = tf.matmul(value, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], matmul_qk.dtype)
    logits = matmul_vk / tf.math.sqrt(depth)
    # add the mask to zero out padding tokens
    mask = tf.cast(mask, logits.dtype)
    logits += (mask * -1e9)
    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    # sns.heatmap(attention_weights.numpy()[0,0,:,:])
    # plt.show()
    output = tf.matmul(attention_weights, query)
    return output



def gelu(x):
    """ Gaussian Error Linear Unit.
    Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + tf.math.erf(x / 1.4142135623730951))
    return x * cdf


def causal_attention_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_padding_mask(x):
    """
    Mark padded positions (token == 0) to mask with 1
    """
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]


def position_range(size):
    return tf.range(size, dtype=tf.int32, name="PositionRange")

