import tensorflow.compat.v2 as tf


def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], matmul_qk.dtype)
    logits = matmul_qk / tf.math.sqrt(depth)
    mask = tf.cast(mask, logits.dtype)
    logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(logits, axis=-1)
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
    mask = tf.cast(mask, logits.dtype)
    logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, query)
    return output


def gelu(x):
    """
    Gaussian Error Linear Unit
    https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + tf.math.erf(x / 1.4142135623730951))
    return x * cdf


def swish(x):
    """
    Swish: Self-Gated Activation Function, discovered via reinforcement learning
    https://arxiv.org/abs/1710.05941
    """
    return x * tf.math.sigmoid(x)


def mish(x):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/abs/1908.08681
    """
    return x * tf.math.tanh(tf.math.log((1 + tf.math.exp(x))))


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

