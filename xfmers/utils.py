import tensorflow.compat.v2 as tf
import tensorflow.keras.backend as K
import time
    

def perplexity(y_true, y_pred):
    y_pred = tf.keras.activations.softmax(y_pred)
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    _perplexity = K.pow(2.0, cross_entropy)
    return _perplexity


def bpc(y_true, y_pred):
    y_pred = tf.keras.activations.softmax(y_pred[:,-1])
    cross_entropy = K.sparse_categorical_crossentropy(y_true[:,-1], y_pred)
    _bpc = cross_entropy / 0.69314718056 # ln2
    return _bpc
    
    
class NoamSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model=512, warmup_steps=4000, scale=1, epsilon=1e-8):
        super(NoamSchedule, self).__init__()
        self.warmup_steps = warmup_steps
        self.k = self.warmup_steps**-1.5
        self.epsilon = epsilon
        self.scale = scale
        self.rsqrt_d_model = self.scale / (d_model**0.5)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.k)
        return self.rsqrt_d_model * tf.math.minimum(arg1, arg2) + self.epsilon

class WarmupExpDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_epochs, flat_epochs, max_epochs, epoch_steps, min_lr=1e-5, decay_exp=5, name="DecayWithWarmup"):
        super(WarmupExpDecay, self).__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_epochs * epoch_steps
        self.flat_steps = flat_epochs * epoch_steps
        self.max_steps = (max_epochs) * epoch_steps
        self.offset = self.max_steps + self.flat_steps
        self.min_lr = min_lr
        self.decay_exp = decay_exp
        self.name = name

    def __call__(self, step):
        with tf.device("/CPU:0"):
            warmup_lr = self.base_lr * step/self.warmup_steps
            flat_lr = self.base_lr
            lr = tf.math.minimum(warmup_lr, flat_lr)
            decay_lr = self.base_lr * ((self.offset-step)/self.max_steps)**self.decay_exp
            lr = tf.math.minimum(lr, decay_lr)
            return lr + self.min_lr
    
    
class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, eg_per_epoch):
        super(TimeHistory, self).__init__()
        self.times = None
        self.eg_per_epoch = eg_per_epoch
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        epoch_duration = time.time() - self.epoch_time_start
        self.times.append(epoch_duration)
        print("\nEg/sec:", int(self.eg_per_epoch/epoch_duration))
