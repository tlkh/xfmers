import json
from .. import ops

class TransformerConfig(object):
    def __init__(self):
        self.name = "Transformer"
        self.model_dim = 768
        self.num_heads = 12
        self.causal_attention = False
        self.attention_mode = "normal"
        self.attention_modes = ["normal", "efficient", "lsh"]
        assert self.attention_mode in self.attention_modes
        self.shared_qk = False
        self.fused_qkv = False
        self.vocab_size = 8192
        self.max_seq_len = 512
        self.embedding_token_scale = 1
        self.embedding_projection = None
        self.ffn_dim = 3072
        self.dropout = 0.1
        self.activation = ops.gelu
        self.conv1d_kernel = 1
        self.conv1d_padding = "same"
        self.epsilon = 1e-6
        self.reversible = False
        self.layers = 6
        self.weight_sharing = False
        
        
    def summary(self):
        pass
    
    def load_from(self, json_path):
        pass
    
    def save_to(self, json_path):
        pass
    
    