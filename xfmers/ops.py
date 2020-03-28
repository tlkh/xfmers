import tensorflow.compat.v2 as tf


def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    mask = tf.cast(mask, tf.float32)
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
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_vk / tf.math.sqrt(depth)
    mask = tf.cast(mask, tf.float32)
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


# ops used for LSH Attention
# referenced from:
# https://github.com/cerebroai/reformers/blob/master/reformers/TFutils.py


def sort_key_val(t1, t2, dim=-1):
    values = tf.sort(t1, axis=dim)
    t2 = tf.broadcast_to(t2, t1.shape)
    return values, tf.gather(t2, tf.argsort(t1, axis=dim), axis=dim)


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return tf.squeeze(tf.gather(values, indices[:, :, None], axis=1))


def make_unit_length(x, epsilon=1e-6):
    norm = tf.norm(x,  ord=2, axis=-1, keepdims=True)
    return tf.math.truediv(x, norm + epsilon)


def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tf.reshape(tensor,  [-1, last_dim])
    summed_tensors = [c.sum(axis=-1) for c in tf.chunk(tensor, chunks, axis=0)]
    return tf.reshape(torch.concat(summed_tensors, axis=0), orig_size)


def process_inputs_chunk(fn, *args, chunks=1):
    chunked_inputs = list(map(lambda x: tf.split(x, chunks, axis=0), args))
    outputs = [fn(*input_pair) for input_pair in zip(*chunked_inputs)]
    return outputs


def look_one_back(x):
    """
    Allow each chunk to attend within itself, and also one chunk back.
    Chunk boundaries might occur in the middle of a sequence of items from the
    same bucket, so this increases the chances of attending to relevant items.
    """
    x_extra = tf.concat([x[:, -1:, ...], x[:, :-1, ...]], axis=1)
    return tf.concat([x, x_extra], axis=2)




