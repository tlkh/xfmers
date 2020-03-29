import tensorflow as tf
from . import config
from .. import ops

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head Attention (MHA) block
    """
    def __init__(self, config, name="MultiHeadAttention"):
        """
        Initialize MHA
        Args:
            d_model: (int) Dimension of hidden layers. Must be a multiple of `num_heads`
            num_heads: (int) Number of attention heads
            causal: (int) Disallow "looking into the future" by masking future time steps
            efficient: (bool) Alternative efficient implementation (https://arxiv.org/abs/1812.01243)
            shared_qk: (bool) Share Query and Key weights 
        Inputs:
            inputs: (dict) Dictionary with keys "query", "key", "value", "mask"
        Outputs:
            N-D tensor with shape: `(batch_size, ..., d_model)`
        """
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        self.causal = causal
        self.efficient = efficient
        self.shared_qk = shared_qk
        assert self.d_model % self.num_heads == 0
        self.depth = self.d_model // self.num_heads
        
        self.query_dense = tf.keras.layers.Dense(units=self.d_model)
        if self.shared_qk:
            self.key_dense = self.query_dense
        else:
            self.key_dense = tf.keras.layers.Dense(units=self.d_model)
        self.value_dense = tf.keras.layers.Dense(units=self.d_model)
        self.final_linear = tf.keras.layers.Dense(units=self.d_model)
        
        if self.efficient:
            self.attention_op = ops.efficient_attention
        else:
            self.attention_op = ops.scaled_dot_product_attention
        
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs,
                            shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs["query"], inputs["key"], inputs["value"], inputs["mask"]
        
        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        # shape: (batch, seq, d_model)

        # split heads
        batch_size = tf.shape(query)[0]
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        # shape: (batch, heads, seq, d_model//heads)

        if self.causal:
            mask_len = tf.shape(mask)[-1]
            causal_mask = ops.causal_attention_mask(mask_len)
            mask += causal_mask

        # scaled dot-product attention
        scaled_attention = self.attention_op(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))
        # final linear layer
        outputs = self.final_linear(concat_attention)
        return outputs
    
    def get_config(self):
        return {"d_model": self.d_model,
                "num_heads": self.num_heads,
                "efficient": self.efficient,
                "shared_qk": self.shared_qk,
                "causal": self.causal}
    

class MultiHeadAttentionFusedQKV(tf.keras.layers.Layer):
    """
    Multi-head Attention (MHA) block with fused QKV operation
    """
    def __init__(self, d_model, num_heads, causal=False, efficient=False, name="MultiHeadAttention"):
        """
        Initialize MHA
        Args:
            d_model: (int) Dimension of hidden layers. Must be a multiple of `num_heads`
            num_heads: (int) Number of attention heads
            causal: (int) Disallow "looking into the future" by masking future time steps
            efficient: (bool) Alternative efficient implementation (https://arxiv.org/abs/1812.01243)
            shared_qk: (bool) Share Query and Key weights 
        Inputs:
            inputs: (dict) Dictionary with keys "query", "key", "value", "mask"
        Outputs:
            N-D tensor with shape: `(batch_size, ..., d_model)`
        """
        super(MultiHeadAttentionFusedQKV, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        self.causal = causal
        self.efficient = efficient
        assert self.d_model % self.num_heads == 0
        self.depth = self.d_model // self.num_heads

        self.fused_qkv = tf.keras.layers.Dense(units=self.d_model*3)
        
        self.final_linear = tf.keras.layers.Dense(units=self.d_model)
        
        if self.efficient:
            self.attention_op = ops.efficient_attention
        else:
            self.attention_op = ops.scaled_dot_product_attention
        
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs,
                            shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs["query"], inputs["key"], inputs["value"], inputs["mask"]
        # shape: (batch, seq, d_model)
        
        # fused qkv operation
        fused_input = tf.concat([query, key, value], axis=-1, name="concat_for_qkv")
        qkv = self.fused_qkv(fused_input)
        query, key, value = tf.split(qkv, num_or_size_splits=3, axis=-1, num=3, name="split_after_qkv")

        # split heads
        batch_size = tf.shape(query)[0]
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        # shape: (batch, heads, seq, d_model//heads)

        if self.causal:
            mask_len = tf.shape(mask)[-1]
            causal_mask = ops.causal_attention_mask(mask_len)
            mask += causal_mask

        # scaled dot-product attention
        scaled_attention = self.attention_op(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))
        # final linear layer
        outputs = self.final_linear(concat_attention)
        return outputs
    
    def get_config(self):
        return {"d_model": self.d_model,
                "num_heads": self.num_heads,
                "efficient": self.efficient,
                "causal": self.causal}

    
class UnsortLogits(tf.keras.layers.Layer):
    def __init__(self):
        super(UnsortLogits, self).__init__()

    def call(self, so, slogits):
        so, slogits = tf.stop_gradient(so), tf.stop_gradient(slogits)
        o = ops.batched_index_select(so, undo_sort)
        _, logits = ops.sort_key_val(sticker, slogits, dim=-1)
        return o, logits
    
    
    
class LSHAttention(tf.keras.Model):
    """
    Referenced from:
    https://github.com/cerebroai/reformers/blob/master/reformers/TFefficient_attention.py
    """
    def __init__(self, dropout=0, bucket_size=64, n_hashes=8, causal=False,
                 allow_duplicate_attention=True, attend_across_buckets=True, rehash_each_round=True,
                 drop_for_hash_rate=0, random_rotations_per_head=False):
        super(LSHAttention, self).__init__()
        assert rehash_each_round or allow_duplicate_attention, (
            "The setting {allow_duplicate_attention=False, rehash_each_round=False} is not implemented.")
        self.dropout_rate = dropout
        self.causal = causal
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size
        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout_for_hash = tf.keras.layers.Dropout(self.dropout_rate)

    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]

        # See: https://arxiv.org/pdf/1509.02897.pdf
        # Sample different random rotation for each round of hashing to decrease the probability of hash miss
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (batch_size if self._random_rotations_per_head else 1,
                           vecs.shape[-1],
                           self.n_hashes if self._rehash_each_round else 1,
                           rot_size // 2)

        random_rotations = tf.broadcast_to(tf.random.normal(rotations_shape),
                                           (batch_size, vecs.shape[-1], self.n_hashes if self._rehash_each_round else 1, rot_size // 2))

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = tf.einsum("btf,bfhi->bhti", dropped_vecs, random_rotations)

        if self._rehash_each_round:
            rotated_vecs = tf.concat([rotated_vecs, -rotated_vecs], axis=-1)
            buckets = tf.math.argmax(rotated_vecs, axis=-1)
            # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
            # bucket numbers from different hashing rounds don"t overlap.
            offsets = tf.range(self.n_hashes)
            offsets = tf.reshape(offsets * n_buckets, (1, -1, 1))
            offsets = tf.cast(offsets, tf.int32)
            buckets = tf.reshape(buckets + offsets, (batch_size, -1,))
        else:
            rotated_vecs = tf.concat([rotated_vecs, -rotated_vecs], axis=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = tf.squeeze(rotated_vecs, axis=0)
            bucket_range = tf.range(rotated_vecs.shape[-1])
            bucket_range = tf.reshape(bucket_range, (1, -1))
            bucket_range = tf.broadcast_to(bucket_range, rotated_vecs.shape)

            _, buckets = ops.sort_key_val(rotated_vecs, bucket_range, axis=-1)
            buckets = buckets[:, -self.n_hashes:]

            h, *_ = buckets.shape 
            buckets = tf.reshape(buckets.permute((*_, h)), (-1,))

        return buckets

    def call(self, qk, v):
        batch_size, seqlen, _ = qk.shape

        n_buckets = seqlen // self.bucket_size
        n_bins = n_buckets

        buckets = self.hash_vectors(n_buckets, qk)
        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen

        ticker = tf.expand_dims(tf.range(self.n_hashes * seqlen), axis=0)
        buckets_and_t = seqlen * buckets + tf.cast((ticker % seqlen), tf.int64)
        buckets_and_t = tf.stop_gradient(buckets_and_t)

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = ops.sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = ops.sort_key_val(sticker, ticker, dim=-1)
        del ticker

        sbuckets_and_t = tf.stop_gradient(sbuckets_and_t)
        sticker = tf.stop_gradient(sticker)
        undo_sort = tf.stop_gradient(undo_sort)

        st = (sticker % seqlen)
        sqk = ops.batched_index_select(qk, st)
        sv = ops.batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        bq_t = bkv_t = tf.reshape(st, (batch_size, self.n_hashes * n_bins, -1))
        bqk = tf.reshape(sqk, (batch_size, self.n_hashes * n_bins, -1, sqk.shape[-1]))
        bv = tf.reshape(sv, (batch_size, self.n_hashes * n_bins, -1, sv.shape[-1]))
        bq_buckets = bkv_buckets = tf.reshape(sbuckets_and_t // seqlen, (batch_size, self.n_hashes * n_bins, -1))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = ops.make_unit_length(bqk)

        bk = ops.look_one_back(bk)
        bv = ops.look_one_back(bv)
        bkv_t = ops.look_one_back(bkv_t)
        bkv_buckets = ops.look_one_back(bkv_buckets)

        # Dot-product attention.
        dots = tf.einsum("bhie,bhje->bhij", bq, bk) * (bq.shape[-1] ** -0.5)

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :] 
            dots = tf.math.multiply(dots, tf.cast(mask, tf.float32)) + (1-tf.cast(mask, tf.float32)) * float("-inf")
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots = tf.math.multiply(dots, tf.cast(self_mask, tf.float32)) + (1-tf.cast(self_mask, tf.float32)) * (- 1e5)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots = tf.math.multiply(dots, tf.cast(bucket_mask, tf.float32)) + (1-tf.cast(bucket_mask, tf.float32)) * float("-inf")
            del bucket_mask

        # Don"t double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % (self.n_hashes * n_bins)
            if not self._attend_across_buckets:
                locs1 = buckets * (self.n_hashes * n_bins) + locs1
                locs2 = buckets * (self.n_hashes * n_bins) + locs2
            locs = tf.transpose(
                tf.concat([
                    tf.reshape(locs1, (batch_size, self.n_hashes, seqlen)),
                    tf.reshape(locs2, (batch_size, self.n_hashes, seqlen)),
                ], 1),
            perm=[0, 2, 1]) 

            slocs = ops.batched_index_select(locs, st)
            b_locs = tf.reshape(slocs, (batch_size, self.n_hashes * n_bins, -1, 2 * self.n_hashes))

            b_locs1 = b_locs[:, :, :, None, :self.n_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, self.n_hashes))
            bq_locs = tf.reshape(bq_locs, b_locs.shape)
            bkv_locs = ops.look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = ops.chunked_sum(dup_counts, chunks=(self.n_hashes * batch_size))
            dup_counts = tf.stop_gradient(dup_counts)
            assert dup_counts.shape == dots.shape
            dots = dots - tf.log(dup_counts + 1e-9)
            del dup_counts

        # Softmax.
        dots_logsumexp = tf.math.reduce_logsumexp(dots, axis=-1, keepdims=True)
        dots = tf.exp(dots - dots_logsumexp)
        dots = self.dropout(dots)

        bo = tf.einsum("buij,buje->buie", dots, bv)
        so = tf.reshape(bo, (batch_size, -1, bo.shape[-1]))
        slogits = tf.reshape(dots_logsumexp, (batch_size, -1,))
            
        unsortlogits = UnsortLogits()
        o, logits = unsortlogits(so, slogits)

        if self.n_hashes == 1:
            out = o
        else:
            o = tf.reshape(o, (batch_size, self.n_hashes, seqlen, o.shape[-1]))
            logits = tf.reshape(logits, (batch_size, self.n_hashes, seqlen, 1))
            probs = tf.exp(logits - tf.math.reduce_logsumexp(logits, axis=1, keepdims=True))
            out = tf.reduce_sum(o * probs, axis=1)

        assert out.shape == v.shape
        return out, buckets
    

class LSHSelfAttention(tf.keras.Model):
    def __init__(self, emb, heads=8, bucket_size=64, n_hashes=8, causal=False, attn_chunks=None, random_rotations_per_head=False, attend_across_buckets=True, allow_duplicate_attention=True):
        super(LSHSelfAttention, self).__init__()
        assert emb % heads == 0, "dimensions must be divisible by number of heads"
        self.emb = emb
        self.heads = heads
        if attn_chunks is None:
            self.attn_chunks = heads 
        else:
            self.attn_chunks = attn_chunks
        self.toqk = tf.keras.layers.Dense(self.emb, use_bias=False)
        self.tov = tf.keras.layers.Dense(self.emb, use_bias=False)
        self.to_out = tf.keras.layers.Dense(self.emb)
        self.bucket_size = bucket_size
        self.causal = causal
        self.random_rotations_per_head = random_rotations_per_head
        self.attend_across_buckets = attend_across_buckets
        self.allow_duplicate_attention = allow_duplicate_attention
        self.lsh_attn = TFLSHAttention(bucket_size=self.bucket_size,
                                       causal=self.causal,
                                       random_rotations_per_head=self.random_rotations_per_head,
                                       attend_across_buckets=self.attend_across_buckets, 
                                       allow_duplicate_attention=self.allow_duplicate_attention)
        
    def merge_heads(v):
        return tf.reshape(tf.transpose(tf.reshape(v, (b, t, h, -1)), perm=[0, 2, 1, 3]), (b * h, t, -1)) 

    def split_heads(v):
        return tf.transpose(tf.reshape(v, (b, t, h, -1)), perm=[0, 2, 1, 3])

    def call(self, inputs):
        b, t, e, h = *inputs.shape, self.heads
        assert t % self.bucket_size == 0, f"Sequence length needs to be divisible by target bucket size - {self.bucket_size}"

        qk = self.toqk(inputs)
        v = self.tov(inputs)
        qk = self.merge_heads(qk)
        v = self.merge_heads(v)
        outputs = ops.process_inputs_chunk(self.lsh_attn, qk, v, chunks=self.attn_chunks)
        attn_out = tf.concat([output for (output, _) in outputs], axis=0)
        out = tf.reshape(self.split_heads(attn_out), (b, t, e))

        return self.to_out(out)
    
    