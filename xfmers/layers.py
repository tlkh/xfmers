import sys
import tensorflow as tf
from . import ops


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, transformer_config, name="MultiHeadAttention"):
        """
        Multi-Head Attention (MHA) Layer
        Inputs:
            inputs: (dict) Dictionary with keys "token_inputs", "mask"
        Outputs:
            N-D tensor with shape: `(batch_size, ..., model_dim)`
        """
        super(MultiHeadAttention, self).__init__(name=name)
        self.tconfig = transformer_config
        self.fused_qkv = self.tconfig.fused_qkv
        self.shared_qk = self.tconfig.shared_qk
        self.model_dim = self.tconfig.model_dim
        self.num_heads = self.tconfig.num_heads
        self.causal = self.tconfig.causal
        self.attention_mode = self.tconfig.attention_mode
        self.initializer_range = self.tconfig.initializer_range
        self.weight_decay = self.tconfig.weight_decay
        
        self.depth = self.model_dim // self.num_heads
        
        if self.attention_mode == "normal":
            self.attention_op = ops.scaled_dot_product_attention
        elif self.attention_mode == "efficient":
            self.attention_op = ops.efficient_attention
        else:
            error_string = "Specified attention mode " + str(self.attention_mode) + " is not supported!"
            raise NotImplementedError(error_string)
        
        if self.fused_qkv:
            # combine q, k, v Dense layers into one Dense layer
            if self.shared_qk:
                # share params for q and k
                qkv_units = self.model_dim*2 
            else:
                qkv_units = self.model_dim*3
            self.fused_qkv = tf.keras.layers.Dense(units=qkv_units,
                                                   kernel_initializer=ops.get_initializer(self.initializer_range),
                                                   kernel_regularizer=tf.keras.regularizers.l2(),
                                                   name="d_fused_qkv")
        else:
            self.query_dense = tf.keras.layers.Dense(units=self.model_dim,
                                                     kernel_initializer=ops.get_initializer(self.initializer_range),
                                                     kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                                     name="d_query")
            if self.shared_qk:
                self.key_dense = self.query_dense
            else:
                self.key_dense = tf.keras.layers.Dense(units=self.model_dim,
                                                       kernel_initializer=ops.get_initializer(self.initializer_range),
                                                       kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                                       name="d_key")
            self.value_dense = tf.keras.layers.Dense(units=self.model_dim,
                                                     kernel_initializer=ops.get_initializer(self.initializer_range),
                                                     kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                                     name="d_value")
        # final linear layer
        self.final_linear = tf.keras.layers.Dense(units=self.model_dim,
                                                  kernel_initializer=ops.get_initializer(self.initializer_range),
                                                  kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                                  name="d_mha_final")
        
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs,
                            shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # inputs shape: (batch, seq, model_dim)
        token_inputs, mask = inputs["token_inputs"], inputs["mask"]
        
        if self.fused_qkv:
            qkv = self.fused_qkv(token_inputs)
            if self.shared_qk:
                query, value = tf.split(qkv, num_or_size_splits=2, axis=-1, num=2, name="split_after_qkv")
                key = query
            else:
                query, key, value = tf.split(qkv, num_or_size_splits=3, axis=-1, num=3, name="split_after_qkv")
        else:
            query = self.query_dense(token_inputs)
            key = self.key_dense(token_inputs)
            value = self.value_dense(token_inputs)

        # split heads
        batch_size = tf.shape(query)[0]
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        # shape: (batch, heads, seq, model_dim//heads)

        if self.causal:
            mask_len = tf.shape(mask)[-1]
            causal_mask = ops.causal_attention_mask(mask_len)
            mask += causal_mask
            
        scaled_attention = self.attention_op(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.model_dim))
        outputs = self.final_linear(concat_attention)
        return outputs
    
    def get_config(self):
        return {"fused_qkv": self.fused_qkv,
                "shared_qk": self.shared_qk,
                "model_dim": self.model_dim,
                "num_heads": self.num_heads,
                "causal": self.causal,
                "attention_mode": self.attention_mode,
                "initializer_range": self.initializer_range,
                "weight_decay": self.weight_decay}

    
class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, transformer_config, name="TransformerLayer"):
        """
        Single Transformer block implementing MHA
        Inputs:
            inputs: (dict) Dictionary with keys "token_inputs" and "mask_inputs"
        Outputs:
            N-D tensor with shape: `(batch_size, ..., model_dim)`
        """
        super(TransformerLayer, self).__init__(name=name)
        self.tconfig = transformer_config
        self.ffn_dim = self.tconfig.ffn_dim
        self.model_dim = self.tconfig.model_dim
        self.dropout = self.tconfig.dropout
        self.activation = self.tconfig.activation
        self.conv1d_kernel = self.tconfig.conv1d_kernel
        self.conv1d_padding = self.tconfig.conv1d_padding
        self.epsilon = self.tconfig.epsilon
        self.initializer_range = self.tconfig.initializer_range
        self.weight_decay = self.tconfig.weight_decay
        
        self.mha = MultiHeadAttention(self.tconfig, name=self.name+"_MultiHeadAttention")
        self.dropout_1 = tf.keras.layers.Dropout(rate=self.dropout)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)
        self.ffn_1 = tf.keras.layers.Conv1D(filters=self.ffn_dim, kernel_size=self.conv1d_kernel,
                                            kernel_initializer=ops.get_initializer(self.initializer_range),
                                            padding=self.conv1d_padding,
                                            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))
        self.act = tf.keras.layers.Activation(self.activation)
        self.ffn_2 = tf.keras.layers.Conv1D(filters=self.model_dim, kernel_size=self.conv1d_kernel,
                                            kernel_initializer=ops.get_initializer(self.initializer_range),
                                            padding=self.conv1d_padding,
                                            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay))
        self.dropout_2 = tf.keras.layers.Dropout(rate=self.dropout)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)
    
    def call(self, inputs):
        token_inputs, mask = inputs["token_inputs"], inputs["mask_inputs"]
            
        mha_output = self.mha({"token_inputs": token_inputs,
                               "mask": mask})
        mha_output = self.dropout_1(mha_output)
        
        mha_residual = self.layer_norm_1(token_inputs + mha_output)
        
        ffn_output = self.ffn_2(self.act(self.ffn_1(mha_residual)))
        ffn_output = self.dropout_2(ffn_output)
        
        outputs = self.layer_norm_2(mha_residual + ffn_output)
            
        return outputs
    
    def get_config(self):
        return {"ffn_dim": self.ffn_dim,
                "model_dim": self.model_dim,
                "dropout": self.dropout,
                "conv1d_padding": self.conv1d_padding,
                "conv1d_kernel": self.conv1d_kernel,
                "activation": self.activation,
                "epsilon": self.epsilon,
                "initializer_range": self.initializer_range,
                "weight_decay": self.weight_decay}
    
    
class TransformerStack(tf.keras.models.Model):
    def __init__(self, transformer_config, name="TransformerStack"):
        """
        Build a stack of Transformer layers as Encoder or Decoder
        Inputs:
            inputs: (dict) Dictionary with keys "token_inputs" and "mask_inputs"
        Outputs:
            N-D tensor with shape: `(batch_size, ..., model_dim)`
        """
        super(TransformerStack, self).__init__(name=name)
        self.tconfig = transformer_config
        self.tlayers = self.tconfig.layers
        self.weight_sharing = self.tconfig.weight_sharing
        
        if self.weight_sharing:
            # create one layer and re-use
            self.xfmer_layer = TransformerLayer(self.tconfig, name=self.name+"_Layer")
            self.xfmer_layers = [self.xfmer_layer for i in range(self.tlayers)]
        else:
            # create a list of layers
            self.xfmer_layers = [
                TransformerLayer(self.tconfig,
                                 name=self.name+"_Layer_"+str(i+1)) for i in range(self.tlayers)
            ]
    
    def call(self, inputs):
        outputs, mask = inputs["token_inputs"], inputs["mask_inputs"]
        
        for layer in self.xfmer_layers:
            outputs = layer({"token_inputs": outputs,
                             "mask_inputs": mask})
            
        return outputs
    
    def get_config(self):
        return {"tlayers": self.tlayers,
                "weight_sharing": self.weight_sharing}
    

class TokenPosEmbedding(tf.keras.layers.Layer):
    """
    Transformer Token and Position Embedding layer
    """
    def __init__(self, transformer_config, name="TokenPosEmbedding"):
        """
        Initialize Transformer Embedding layer for token and position embeddings.
        This must be the very first layer in any model after the Input layer.
        Inputs:
            token_inputs: 1D Tensor of integer numbers (< vocab_size) and of maximum length max_seq_len
        Outputs:
            N-D tensor with shape: `(batch_size, ..., model_dim)`
        """
        super(TokenPosEmbedding, self).__init__(name=name)
        self.tconfig = transformer_config
        self.vocab_size = self.tconfig.vocab_size
        self.model_dim = self.tconfig.model_dim
        self.max_seq_len = self.tconfig.max_seq_len
        self.embedding_token_scale = self.tconfig.embedding_token_scale
        self.embedding_projection = self.tconfig.embedding_projection
        self.initializer_range = self.tconfig.initializer_range
        self.weight_decay = self.tconfig.weight_decay
        self.epsilon = self.tconfig.epsilon
        
        self.token_embeddings = tf.keras.layers.Embedding(self.vocab_size,
                                                          self.model_dim,
                                                          embeddings_initializer=ops.get_initializer(self.initializer_range),
                                                          name="token_embedding",)
        self.pos_embeddings = tf.keras.layers.Embedding(self.max_seq_len,
                                                        self.model_dim,
                                                        embeddings_initializer=ops.get_initializer(self.initializer_range),
                                                        name="token_embedding",)
        if self.embedding_projection:
            self.projection = tf.keras.layers.Dense(units=self.embedding_projection,
                                                    kernel_initializer=ops.get_initializer(self.initializer_range),
                                                    kernel_regularizer=tf.keras.regularizers.l2(self.tconfig.weight_decay))
            
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)

    def call(self, inputs):
        pos_range = ops.position_range(tf.shape(inputs)[-1])

        self.token_vector = self.token_embeddings(inputs)
        if self.embedding_token_scale != 1:
            self.token_vector *= self.embedding_token_scale
        self.pos_vector = self.pos_embeddings(pos_range)

        embeddings = self.token_vector + self.pos_vector
        
        if self.embedding_projection:
            embeddings = self.projection(embeddings)
            
        embeddings = self.layer_norm(embeddings)
        
        return embeddings
    
    def get_config(self):
        return {"vocab_size": self.vocab_size,
                "model_dim": self.model_dim,
                "max_seq_len": self.max_seq_len,
                "embedding_token_scale": self.embedding_token_scale,
                "embedding_projection": self.embedding_projection,
                "initializer_range": self.initializer_range,
                "weight_decay": self.weight_decay,
                "epsilon": self.epsilon}

    
class PaddingMaskGenerator(tf.keras.layers.Layer):
    def __init__(self, name="PaddingMaskGenerator"):
        """
        Creates a mask for any padding (`0`) characters in the input sequence
        Inputs:
            token_inputs: 1D Tensor of integer numbers
        Outputs:
            N-D tensor
        """
        super(PaddingMaskGenerator, self).__init__(name=name)
        
    def call(self, inputs):
        return ops.create_padding_mask(inputs)
    
    def get_config(self):
        return {}
    
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
            # In this transformer_configuration, we map each item to the top self.n_hashes buckets
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
    
    