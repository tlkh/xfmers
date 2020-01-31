MAX_SEQ_LEN = 128
BATCH_SIZE = 32
BUFFER_SIZE = 1000000
NUM_LAYERS = 3
NUM_HEADS = 12
D_MODEL = NUM_HEADS * 32
UNITS = D_MODEL * 16
DROPOUT = 0.1
WEIGHT_SHARING = False

import os

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["NCCL_DEBUG"] = "WARN"

import tensorflow.compat.v2 as tf
import horovod.tensorflow.keras as hvd

hvd.init()
hvd_rank = hvd.rank()
hvd_size = hvd.size()

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[hvd_rank], 'GPU')

import os
import re
import numpy as np
from tqdm import tqdm, trange

import tensorflow_datasets as tfds
from xfmers import models
from xfmers import utils

tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

def load_text8(text8_path, num_char=None):
    with open(text8_path, 'r', encoding="utf8") as file:
        data = file.read().replace('\n', '').strip()
    if num_char:
        data = data[:num_char]
    return data.strip()


text_8_train = load_text8("./data/train.txt.raw")
char_set = list(set(list(text_8_train)))
char_set.sort()
print("Character set:", char_set)
len_corpus = len(text_8_train)

shard_len = len_corpus//hvd_size - 1

text_8_train = text_8_train[hvd_rank*shard_len:(hvd_rank+1)*shard_len]
print(hvd_rank, "Loaded", len(text_8_train), "characters into training set")

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=len(char_set)+1, char_level=True)
tokenizer.fit_on_texts(["".join(char_set)])

# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.num_words], [tokenizer.num_words + 1]

# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.num_words + 2

print("Vocab size:", VOCAB_SIZE)

# pad to 8
vocab_size_mult = VOCAB_SIZE // 8
vocab_size_r = VOCAB_SIZE % 8
if vocab_size_r > 0:
    VOCAB_SIZE = (vocab_size_mult + 1) * 8
    print("Vocab size padded to:", VOCAB_SIZE, "(performance reasons)")

# Tokenize, filter and pad sentences
def prepare_sequences(corpus, tokenizer, seq_len=512):
    print("Encoding corpus...", corpus[:10])
    corpus = tokenizer.texts_to_sequences([corpus])[0]
    total_words = len(corpus)
    print("Done! Token count:", total_words)
    print(corpus[:10])
    
    print("Generating sequences...")
    list_seq = []
    for i in trange(0, total_words-seq_len, seq_len//32):
        seq = START_TOKEN + corpus[i:i+seq_len]
        list_seq.append(seq)
    print("Done!")

    return np.asarray(list_seq, dtype="int")

list_seq = prepare_sequences(text_8_train, tokenizer, MAX_SEQ_LEN)

# free some memory
del text_8_train

text_8_val = load_text8("./data/valid.txt.raw")

len_corpus = len(text_8_val)
shard_len = len_corpus//hvd_size - 1
text_8_val = text_8_val[hvd_rank*shard_len:(hvd_rank+1)*shard_len]

list_seq_val = prepare_sequences(text_8_val, tokenizer, MAX_SEQ_LEN)

del text_8_val

print("Training sequences:", list_seq.shape)
print("Validation sequences:", list_seq_val.shape)

# training

dataset = tf.data.Dataset.from_tensor_slices(({'inputs': list_seq[:,:-1]},
                                              {'outputs': list_seq[:,-1]},))
dataset = dataset.repeat(-1)
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(8)
_ = dataset.take(1)

# validation

val_dataset = tf.data.Dataset.from_tensor_slices(({'inputs': list_seq_val[:,:-1]},
                                                  {'outputs': list_seq_val[:,-1]},))
val_dataset = val_dataset.repeat(-1)
val_dataset = val_dataset.shuffle(BUFFER_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE*2)
val_dataset = val_dataset.prefetch(8)
_ = val_dataset.take(1)

model = models.DecoderTransformer(
    vocab_size=VOCAB_SIZE,
    dec_layers=NUM_LAYERS,
    ff_units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT,
    max_seq_len=MAX_SEQ_LEN,
    weight_sharing=WEIGHT_SHARING
)

if hvd_rank == 0:
    tf.keras.utils.plot_model(
        model,
        to_file='model.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
    )

#model.load_weights("checkpoint-5-1.18.h5")

opt = tf.keras.optimizers.Adam(0.00001)
opt = hvd.DistributedOptimizer(opt,
                               compression=hvd.Compression.fp16,
                               sparse_as_dense=True)
opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")


model.compile(optimizer=opt,
              loss="sparse_categorical_crossentropy",
              metrics=[utils.bpc],
              experimental_run_tf_function=False)
model.run_eagerly = False

callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    #hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=10, verbose=1),
]

if hvd_rank == 0:
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    from nvstatsrecorder.callbacks import NVStats
    nv_stats = NVStats(gpu_index=0, interval=10)
    model.summary()
    verbose = 1
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}-{loss:.2f}.h5', save_weights_only=True))
    callbacks.append(nv_stats)
else:
    verbose = 0
    
train_steps = list_seq.shape[0]//BATCH_SIZE
print(hvd_rank, "training steps:", train_steps)

print(hvd_rank, "Starting training:")
    
model.fit(dataset, callbacks=callbacks,
          epochs=10, steps_per_epoch=21634, # 21634
          verbose=verbose)

if hvd_rank == 0:
    SMOOTH = 10
    nv_stats_recorder = nv_stats.recorder
    nv_stats_recorder.plot_gpu_util(smooth=SMOOTH, show=False, outpath="gpu_util.png")
    model.save_weights("model.h5")
    