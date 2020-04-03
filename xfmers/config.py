import json
from . import ops


class TransformerConfig(object):
    def __init__(self, model_dim=768, num_heads=12, layers=6, ffn_dim=3072,
                 causal=False, attention_mode="normal", shared_qk=False, fused_qkv=False,
                 vocab_size=8192, max_seq_len=512, embedding_token_scale=1,
                 dropout=0.1, activation=ops.gelu, epsilon=1e-6, initializer_range=0.02,
                 conv1d_kernel=1, conv1d_padding="same", weight_sharing=False,
                 weight_decay=1e-6):
        self.name = "Transformer"
        self.model_dim = model_dim
        self.num_heads = num_heads
        assert self.model_dim % self.num_heads == 0
        self.ffn_dim = ffn_dim
        self.layers = layers
        self.causal = causal
        self.attention_mode = attention_mode
        self._attention_modes = ["normal", "efficient", "lsh"]
        assert self.attention_mode in self._attention_modes
        self.shared_qk = shared_qk
        self.fused_qkv = fused_qkv
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_token_scale = embedding_token_scale
        self.embedding_projection = None
        self.dropout = dropout
        self.activation = activation
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_padding = conv1d_padding
        self.epsilon = epsilon
        self.weight_sharing = weight_sharing
        self.initializer_range = initializer_range
        self.weight_decay = weight_decay
        
        
    def summary(self):
        title = "Model: " + self.name
        print("")
        print(title)
        print("="*len(title))
        print("    Layers:", self.layers)
        print(" Model Dim:", self.model_dim)
        print("Attn Heads:", self.num_heads)
        print("="*len(title))
        print("")
    
    
    def load(self, json_path):
        pass
    
    
    def save(self, json_path):
        with open(json_path, 'w') as outfile:
            json.dump(self.__dict__, outfile)
    
    