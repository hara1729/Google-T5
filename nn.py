import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import pathlib
import sys
import warnings
from termcolor import cprint
from time import sleep

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

from dataloader import WikiQARAGLoader, SquadRAGLoader
from archs import T5Model

os.system('clear')
warnings.filterwarnings('ignore')


def set_precision(precision = 'float32'):
    """
    Set the precision for model training. Supported values are 'float32' and 'float16'.

    Args:
        precision (str): Precision mode to use. Options: 'float32', 'float16'.

    Raises:
        ValueError: If an unsupported precision mode is passed.
    """
    if precision == 'float16':
        mixed_precision.set_global_policy('mixed_float16')
        print("\u2705 Precision set to float16 (Mixed Precision Enabled)")
    elif precision == 'bfloat16':
        mixed_precision.set_global_policy('mixed_bfloat16')
        print("\u2705 Precision set to bfloat16 (Mixed Precision Enabled)")
    elif precision == 'float32':
        mixed_precision.set_global_policy('float32')
        print("\u2705 Precision set to float32 (Default Precision)")
    else:
        raise ValueError(f"Unsupported precision mode: {precision}. Use 'float16', 'bfloat16' or 'float32'.")

def sequence_xent_ignore_pad(pad_id: int, vocab_size: int, label_smoothing: float):
    '''
    Cross-entropy loss between labels and logits. Uses label smoothing to avoid NaNs and Inf losses
    early in the training. In order to use label smoothing we use `CategoricalCrossentopy` loss which
    support `label_smoothing` via its keras API.
    '''
    cce = keras.losses.CategoricalCrossentropy(from_logits = True, 
                                               label_smoothing = label_smoothing, 
                                               reduction = keras.losses.Reduction.NONE)
    
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
        y_true_ohe = tf.one_hot(tf.cast(y_true, tf.int32), depth = vocab_size, dtype = y_pred.dtype)

        per_token = cce(y_true_ohe, y_pred) * mask
        
        return tf.reduce_sum(per_token) / (tf.reduce_sum(mask) + 1e-3)
    
    return loss

class NaNCallback(keras.callbacks.Callback):
    '''
    Checks for NaN/Inf in input data to model if loss becomes Nan.
    Utility function. Not used.
    '''
    def __init__(self, train_data, **kwargs):
        super().__init__(**kwargs)
        self.train_data = train_data
    
    def on_train_batch_end(self, batch, logs = None):
        if np.isinf(logs['loss']):
            current_batch = next(iter(self.train_data.skip(batch).take(1)))
        
            (encoder_tokens, decoder_tokens), labels = current_batch
            encoder_tokens = encoder_tokens.numpy()
            decoder_tokens = decoder_tokens.numpy()
            labels = labels.numpy()
            
            print(f"encoder tokens (Nan/Inf): {~np.any(np.isfinite(encoder_tokens))}")
            print(f"decoder tokens (Nan/Inf): {~np.any(np.isfinite(decoder_tokens))}")
            print(f"labels (Nan/Inf): {~np.any(np.isfinite(labels))}")
        
            sys.exit("Found Inf in loss")


class SaveBestCheckpoint(keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def on_train_begin(self, logs = None):
        self.best_accuracy = 0
    
    def on_epoch_end(self, epoch, logs = None):
        val_accuracy = logs.get('val_accuracy', None)
        if val_accuracy is not None:
            if val_accuracy > self.best_accuracy:
                self.model.t5_model.save("./saved_models/t5-1.keras")
                pathlib.Path("t5_kwargs.json").write_text(json.dumps(self.model.t5_kwargs))
                self.best_accuracy = val_accuracy
                print()
                print(f"val accuracy improved to {val_accuracy:.4f}. Saved checkpoint to: ./saved_models/t5-1.keras")


@keras.utils.register_keras_serializable(package = "Custom")
class WarmUpCosineDecayRestart(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, 
                 initial_learning_rate,
                 warmup_steps,
                 first_decay_steps,
                 t_mul = 2.0,
                 m_mul = 1.0,
                 alpha = 0.0,
                 name = None):
        
        super().__init__()
        
        self.name = name or "WarmUpCosineDecayRestart"
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        self.cosine_decay = keras.optimizers.schedules.CosineDecayRestarts(
                                                                                initial_learning_rate = initial_learning_rate,
                                                                                first_decay_steps = first_decay_steps,
                                                                                t_mul = t_mul,
                                                                                m_mul = m_mul,
                                                                                alpha = alpha
                                                                          )

    def __call__(self, step):
        warmup_lr = self.initial_learning_rate * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
        cosine_lr = self.cosine_decay(step - self.warmup_steps)
        return tf.cond(step < self.warmup_steps,
                       lambda: warmup_lr,
                       lambda: cosine_lr)
        
    def get_config(self):
        config = dict()
        config.update({
                        "initial_learning_rate"   : self.initial_learning_rate,
                        "warmup_steps"            : self.warmup_steps,
                        "first_decay_steps"       : self.first_decay_steps,
                        "t_mul"                   : self.t_mul,
                        "m_mul"                   : self.m_mul,
                        "alpha"                   : self.alpha,
                        "name"                    : self.name
                      })
        
        return config


class T5Trainer(keras.Model):
    '''
    Trainer module for T5 model.
    Methods:
        -- __init__(pad_id, vocab_size, t5_kwargs)
        -- build(input_shape)
        -- call(encoder_tokens, decoder_tokens, training: Optional[bool] = None)
        -- compile(optimizer, **kwargs)
        -- metrics --> property (used only in validation call)
        -- train_step(data)
        -- test_step(data)
    '''
    def __init__(self, pad_id, vocab_size, t5_kwargs, **kwargs):
        '''
        Args:
            -- pad_id: tokenizer pad token id. Used to mask the pad_tokens
            -- vocab_size: tokenizer vocabulary size.
            -- t5_kwargs: key word arguments used construct a T5 model
        '''
        super().__init__(**kwargs)
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.t5_kwargs = t5_kwargs
        self.t5_model = T5Model(**t5_kwargs)
    
    def build(self, input_shape):
        self.t5_model.build(input_shape)
        super().build(input_shape)
    
    def call(self, inputs, training = None):
        '''
        Wrapper around `t5_model.call()`.
        Must be present if subclassing from `keras.Model` in order to be able to serialize the model.
        '''
        encoder_tokens = inputs[0]
        decoder_tokens = inputs[1]
        return self.t5_model([encoder_tokens, decoder_tokens], training = training)

    def compile(self, optimizer, **kwargs):
        '''
        Adds loss function and validation accuract tracker to the model. 
        Must be called before calling `build`.
        '''
        self.loss_fn = sequence_xent_ignore_pad(self.pad_id, self.vocab_size, label_smoothing = 1e-3)
        self.train_acc_tracker = keras.metrics.SparseTopKCategoricalAccuracy(k = 3, name = 'batch_accuracy')
        self.val_acc_tracker = keras.metrics.SparseTopKCategoricalAccuracy(k = 3, name = 'accuracy')
        
        super().compile(optimizer = optimizer, loss = self.loss_fn, **kwargs)
        
   
    def train_step(self, data):
        '''
        Basic training loop.
            1. A forward pass
            2. Calculate loss
            3. Calculate gradients from loss
            4. Apply gradients
        
        The train_step is overridden inorder to clip the logits to a reasonable range [-25, 25].
        This also allows for flexibility to add debugging checks.
        
        We don't monitor training accuracy as it becomes NaN after a certain amount of batches have passed but
        the loss keeps decreasing which suggest the training is proceeding well. This issue happens only while
        training in multi-GPU environment which suggests a potential **BUG** in keras metric accumulation.
        '''
        (enc_tokens, dec_tokens), labels = data
        with tf.GradientTape() as tape:
            logits = self([enc_tokens, dec_tokens], training = True)
            # print_op = tf.print("logits", logits)
            # with tf.control_dependencies([print_op]):
            logits = tf.clip_by_value(logits, -25.0,  25.0)
            with tf.control_dependencies([tf.debugging.check_numerics(logits, "logits")]):
                loss   = self.compiled_loss(
                                                labels,
                                                logits,
                                                regularization_losses = self.losses,
                                            )
                
            scaled_loss = self.optimizer.scale_loss(loss)
        
        scaled_grads = tape.gradient(scaled_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(scaled_grads, self.trainable_variables))
        
        logits = tf.cast(logits, tf.float32)
        mask = tf.cast(tf.not_equal(labels, self.pad_id), logits.dtype)
        self.train_acc_tracker.update_state(labels, logits, sample_weight = mask)
        
        ret_dict = {'loss': loss}
        ret_dict.update({self.train_acc_tracker.name: self.train_acc_tracker.result()})
        
        # reset states manually
        self.train_acc_tracker.count.assign(0)
        self.train_acc_tracker.total.assign(0)
             
        return ret_dict
    
    def test_step(self, data):
        '''
        Validation/Evaluation/Test logic.
        Same as training logic apart from back-propagation and Accuracy calculation.
        '''
        (enc_tokens, dec_tokens), labels = data
        
        logits = self([enc_tokens, dec_tokens], training = False)
        loss = self.loss_fn(labels, logits)
        
        logits = tf.cast(logits, tf.float32)
        mask = tf.cast(tf.not_equal(labels, self.pad_id), logits.dtype)
        self.val_acc_tracker.update_state(labels, logits, sample_weight = mask)
        
        ret_dict = {'loss': loss}
        ret_dict.update({self.val_acc_tracker.name: self.val_acc_tracker.result()})

        
        return ret_dict

def get_combined_dataset(seq_len, batch_size, num_wiki_sentences):
    wiki_loader = WikiQARAGLoader(seq_len = seq_len, batch_size = batch_size, num_sentences = num_wiki_sentences)
    squad_loader = SquadRAGLoader(seq_len = seq_len, batch_size = batch_size)
    
    vocab_size = wiki_loader.tokenizer.vocab_size       # <-- same as squad_loader.tokenizer.vocab_size
    pad_id = wiki_loader.pad_id                         # <-- same as squad_loader.pad_id
    
    cprint(f"Loading train split", "white")
    squad_train = squad_loader.load_split('train')
    wiki_train = wiki_loader.load_split('train')
    
    cprint(f"Loading val split", "yellow")
    squad_val = squad_loader.load_split('validation')
    wiki_val = wiki_loader.load_split('validation')
    
    cprint(f"Dataset created. Shuffling...", "cyan")
    train_dataset = wiki_train.concatenate(squad_train).shuffle(15_000).prefetch(tf.data.AUTOTUNE)
    val_dataset = wiki_val.concatenate(squad_val).shuffle(15_000).prefetch(tf.data.AUTOTUNE)
    cprint(f"Shuffled", "green")
    sleep(1)
    os.system("clear")
    
    return (train_dataset, val_dataset), vocab_size, pad_id


def train():
    precision = "float16"
    seq_len = 256
    
    set_precision(precision)                        # <-- Set mixed precision

    # Load WikiQA Data
    (train_ds, val_ds), vocab_size, pad_id = get_combined_dataset(seq_len = seq_len, batch_size = 192, num_wiki_sentences = 10)
    
    strategy = tf.distribute.MirroredStrategy()     # <-- Distributes work across all GPUs.
    
    with strategy.scope():
        # create a T5 model
        t5_kwargs = dict(
                            vocab_size                  = vocab_size,
                            seq_len                     = seq_len,
                            d_model                     = 768,
                            num_heads                   = 12,
                            d_ff                        = 3072,
                            num_layers                  = 12,
                            attention_type              = "mha",
                            num_query_groups            = None,
                            use_relative_position_bias  = True,
                            attn_dropout                = 0.1,
                            ffn_dropout                 = 0.1,
                            layer_norm_epsilon          = 1e-6
                        )
        
        # pass it to the trainer
        trainer = T5Trainer(pad_id = pad_id, vocab_size = vocab_size, t5_kwargs = t5_kwargs)
        # create lr schedule
        lr_schedule = WarmUpCosineDecayRestart(
                                                    initial_learning_rate = 2e-4,      # “peak” LR after warm-up
                                                    warmup_steps          = 1000,
                                                    first_decay_steps     = 2000,
                                                    t_mul                 = 2.0,       # period doubles every restart
                                                    m_mul                 = 1.0,       # no amplitude shrink each time
                                                    alpha                 = 0.1        # decay to 10% at the end of cycle
                                              )
        # create an AdamW optimizer with the lr_schedule
        optimizer = keras.optimizers.AdamW(
                                            learning_rate = lr_schedule,
                                            weight_decay  = 1e-2,
                                            clipnorm      = 1.0,
                                          )
        # optionally wrap the optimizer in mixed_precision.LossScaleOptimizer if mixed precision is used.
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        # compile the trainer (creates loss function and accuracy metric)
        trainer.compile(optimizer = optimizer)
        
        # build the model by calling it (creates model weights). This must be done POST compiling in keras 3.
        _ = trainer([tf.random.normal(shape = (1, seq_len)), tf.random.normal(shape = (1, seq_len))], training = False)

    trainer.fit(train_ds, validation_data = val_ds, epochs = 30, callbacks = [NaNCallback(train_ds), SaveBestCheckpoint()])
    
    
if __name__ == "__main__":
    devices = tf.config.list_physical_devices('GPU')
    for dev in devices:
        tf.config.experimental.set_memory_growth(dev, True)
    
    train()