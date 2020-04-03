import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fused_qkv", action="store_true", default=True)
parser.add_argument("--weight_sharing", action="store_true", default=False)
parser.add_argument("--layers", default=12, type=int)
parser.add_argument("--heads", default=12, type=int)
parser.add_argument("--steps", type=int)
parser.add_argument("--num_chars", type=int)
parser.add_argument("--outdir", default="./", type=str)
args = parser.parse_args()

import os
import multiprocessing
n_cores = multiprocessing.cpu_count()
os.environ["TF_DISABLE_NVTX_RANGES"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = str(n_cores)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
OUTDIR = args.outdir
if OUTDIR[-1] != "/":
    OUTDIR = OUTDIR + "/"
os.makedirs(OUTDIR, exist_ok=True)
import time
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
tf.config.threading.set_inter_op_parallelism_threads(n_cores)

from xfmers import models
from xfmers import utils
from xfmers import ops
from xfmers import config

MAX_SEQ_LEN = 80
BATCH_SIZE = 80
EPOCHS = 20
BUFFER_SIZE = 10000000
NUM_HEADS = args.heads

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))
BATCH_SIZE *= strategy.num_replicas_in_sync
print("Batch size:", BATCH_SIZE)

def load_text8(text8_path, num_char=None):
    with open(text8_path, "r", encoding="utf8") as file:
        data = file.read().replace("\n", " ").strip()
        data = data.replace("  ", " ")
    if num_char:
        data = data[:num_char]
    return data.strip()

text_8_train = load_text8("./data/train.txt.raw", args.num_chars)

if os.path.isfile("./subwords/text8vocab.subwords"):
    tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file("./subwords/text8vocab")
else:
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus([text_8_train],
                                                                        target_vocab_size=5120)
    tokenizer.save_to_file("./subwords/text8vocab")
if tokenizer.vocab_size%8 == 0:
    VOCAB_SIZE = tokenizer.vocab_size
else:
    VOCAB_SIZE = tokenizer.vocab_size
    mult_8 = VOCAB_SIZE//8
    VOCAB_SIZE = (mult_8 + 1) * 8
    
model_config = config.TransformerConfig(num_heads=NUM_HEADS,
                                        model_dim=NUM_HEADS*64,
                                        layers=args.layers,
                                        ffn_dim=4*NUM_HEADS*64,
                                        causal=True,
                                        vocab_size=VOCAB_SIZE,
                                        shared_qk=False,
                                        fused_qkv=args.fused_qkv,
                                        weight_sharing=args.weight_sharing,
                                        max_seq_len=MAX_SEQ_LEN,
                                        weight_decay=0.001,
                                        dropout=0.1)

model_config.summary()

"""
char_set = list(set(list(text_8_train)))
char_set.sort()
print("Character set:", char_set)
len_corpus = len(text_8_train)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=len(char_set), char_level=True)
tokenizer.fit_on_texts(["".join(char_set)])

VOCAB_SIZE = tokenizer.num_words + 1
"""

print("Vocab size:", VOCAB_SIZE)

def prepare_sequences(corpus, tokenizer, interval=1, seq_len=512):
    """Tokenize, filter and pad sentences"""
    print("Encoding corpus...")
    print(corpus[:MAX_SEQ_LEN])
    corpus = tokenizer.encode(corpus)
    #corpus = tokenizer.texts_to_sequences([corpus])[0]
    print(corpus[:MAX_SEQ_LEN])
    total_words = len(corpus)
    print("Done! Token count:", total_words)
    print("Generating sequences...")
    list_seq = []
    for i in trange(0, total_words-seq_len, interval):
        seq = corpus[i:i+seq_len+1]
        list_seq.append(seq)
    print("Done!")
    return np.asarray(list_seq, dtype="int")

print("Building input pipelines...")
st = time.time()

interval = int(MAX_SEQ_LEN/3)

list_seq = prepare_sequences(text_8_train, tokenizer, interval=interval, seq_len=MAX_SEQ_LEN)

print("Training sequences:", list_seq.shape)

text_8_val = load_text8("./data/valid.txt.raw", args.num_chars)

list_seq_val = prepare_sequences(text_8_val, tokenizer, interval=interval, seq_len=MAX_SEQ_LEN)

print("Validation sequences:", list_seq_val.shape)

# training

dataset = tf.data.Dataset.from_tensor_slices(({"inputs": list_seq[:,:-1]},
                                              {"outputs": list_seq[:,1:]},))
dataset = dataset.repeat(-1)
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.prefetch(32)

train_steps = int(list_seq.shape[0]/BATCH_SIZE)
if args.steps:
    train_steps = args.steps
    
print("Train for", train_steps)

for batch in dataset.take(1):
    x, y = batch
    print("* Input:", x["inputs"].numpy().shape)
    print("* Label:", y["outputs"].numpy().shape)

# validation

val_dataset = tf.data.Dataset.from_tensor_slices(({"inputs": list_seq_val[:,:-1]},
                                                 {"outputs": list_seq_val[:,1:]},))
val_dataset = val_dataset.repeat(-1)
val_dataset = val_dataset.batch(BATCH_SIZE*2, drop_remainder=False)
val_dataset = val_dataset.prefetch(32)

valid_steps = int(list_seq_val.shape[0]/BATCH_SIZE/2)
if args.steps:
    valid_steps = args.steps
    
print("Valid for", valid_steps)

for batch in val_dataset.take(1):
    x, y = batch
    print("* Input:", x["inputs"].numpy().shape)
    print("* Label:", y["outputs"].numpy().shape)

et = time.time()
print("Done in", int(et-st), "seconds")

del text_8_train
del text_8_val

with strategy.scope():
    model = models.Transformer(model_config)
    learning_rate = utils.WarmupExpDecay(
        epoch_steps=train_steps,
        base_lr=1e-5,
        min_lr=1e-8,
        decay_exp=10,
        warmup_epochs=1,
        flat_epochs=EPOCHS-3,
        max_epochs=EPOCHS,
    )
    plt.plot(learning_rate(tf.range(train_steps*(EPOCHS-1), dtype=tf.float32)))
    plt.axvline(train_steps, linestyle="--",  color="g", label="epoch 1")
    plt.title("Learning Rate Schedule")
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.legend()
    plt.show()
    plt.savefig(OUTDIR+"lr_schedule.jpg")
    opt = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-8)
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
    lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #acc = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
    model.compile(optimizer=opt,
                  loss=lossfn,
                  metrics=[utils.bpc])
    model.run_eagerly = False

model.summary()

time_history = utils.TimeHistory(eg_per_epoch=train_steps*BATCH_SIZE)
checkpoint = tf.keras.callbacks.ModelCheckpoint(OUTDIR+"weights.h5", monitor="val_loss", verbose=1,
                                                save_best_only=True,save_weights_only=True)
callbacks = [time_history, checkpoint, ]

with strategy.scope():
    st = time.time()
    history = model.fit(dataset, epochs=EPOCHS, steps_per_epoch=train_steps,
                        validation_data=val_dataset, validation_steps=valid_steps,
                        verbose=1, callbacks=callbacks)
    et = time.time()

df = pd.DataFrame(history.history)
df["time"] = time_history.times
df.to_csv(OUTDIR+"history.csv")
print("Finished in", int(et-st), "seconds")
epoch_time = min(time_history.times)
tokens_per_epoch = BATCH_SIZE*train_steps*MAX_SEQ_LEN
print("Tokens/sec:", int(tokens_per_epoch/epoch_time))

