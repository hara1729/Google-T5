import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

from dataloader import WikiQARAGLoader
from archs import T5Model

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
    cce = keras.losses.CategoricalCrossentropy(from_logits = True, 
                                               label_smoothing = label_smoothing, 
                                               reduction = keras.losses.Reduction.NONE)
    
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, pad_id), y_pred.dtype)
        y_true_ohe = tf.one_hot(y_true, depth = vocab_size, dtype = y_pred.dtype)

        per_token = cce(y_true_ohe, y_pred) * mask
        
        return tf.reduce_sum(per_token) / (tf.reduce_sum(mask) + 1e-3)
    
    return loss

class NaNCallback(keras.callbacks.Callback):
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
            

class T5Trainer(keras.Model):
    def __init__(self, pad_id, vocab_size, t5_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.t5_model = T5Model(**t5_kwargs)
    
    def build(self, input_shape):
        self.t5_model.build(input_shape)
        super().build(input_shape)
    
    def call(self, encoder_tokens, decoder_tokens, training = None):
        return self.t5_model([encoder_tokens, decoder_tokens], training = training)

    def compile(self, optimizer, **kwargs):
        self.loss_fn = sequence_xent_ignore_pad(self.pad_id, self.vocab_size, label_smoothing = 1e-3)
        self.acc_tracker = keras.metrics.SparseCategoricalAccuracy(name = 'accuracy')
        
        super().compile(optimizer = optimizer, **kwargs)
        
    @property
    def metrics(self):
        return [self.acc_tracker]
    
    def train_step(self, data):
        (enc_tokens, dec_tokens), labels = data
        with tf.GradientTape() as tape:
            logits = self(enc_tokens, dec_tokens, training = True)
            mask = tf.cast(tf.not_equal(labels, self.pad_id), logits.dtype)
            logits = tf.clip_by_value(logits, -25.0,  25.0)
            with tf.control_dependencies([tf.debugging.check_numerics(logits, "logits contain Nan/Inf")]):
                loss = self.loss_fn(labels, logits)
        
        grads = tape.gradient(loss, self.t5_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.t5_model.trainable_variables))
        
        ret_dict = {'loss': loss}        
        return ret_dict
    
    def test_step(self, data):
        (enc_tokens, dec_tokens), labels = data
        
        logits = self(enc_tokens, dec_tokens, training = False)
        loss = self.loss_fn(labels, logits)
        
        mask = tf.cast(tf.not_equal(labels, self.pad_id), logits.dtype)
        
        self.acc_tracker.update_state(labels, logits, sample_weight = mask)
        
        ret_dict = {'loss': loss}
        ret_dict.update({m.name: m.result() for m in self.metrics})
        
        return ret_dict


def train():
    loader = WikiQARAGLoader(seq_len = 512, batch_size = 64, num_sentences = 3)
    
    precision = "float32"
    
    # set_precision(precision)

    train_ds = loader.load_split("train")
    val_ds   = loader.load_split("validation")
    
    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        t5_kwargs = dict(
                            vocab_size = loader.tokenizer.vocab_size,
                            seq_len    = 512,
                            d_model    = 256,
                            num_heads  = 4,
                            d_ff = 1024,
                            num_layers = 6,
                            attention_type = "mha",
                            num_query_groups = None,
                            use_relative_position_bias = True,
                            attn_dropout = 0.1,
                            ffn_dropout  = 0.1,
                            layer_norm_epsilon = 1e-6
                        )
        
        trainer = T5Trainer(pad_id = loader.pad_id, vocab_size = loader.tokenizer.vocab_size, t5_kwargs = t5_kwargs)
        optimizer = keras.optimizers.AdamW(
                                            learning_rate = 1e-4,
                                            weight_decay  = 1e-2,
                                            clipnorm      = 5.0,
                                          )
        trainer.compile(optimizer = optimizer)
        trainer.build(input_shape = ((loader.batch_size, loader.seq_len, loader.tokenizer.vocab_size),
                                     (loader.batch_size, loader.seq_len, loader.tokenizer.vocab_size)))

        if precision != "float32":
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    # trainer.fit(train_ds, validation_data = val_ds, epochs = 3, callbacks = [NaNCallback(train_ds)])
    trainer.fit(train_ds, validation_data = val_ds, epochs = 10, callbacks = None)
    
if __name__ == "__main__":
    devices = tf.config.list_physical_devices('GPU')
    for dev in devices:
        tf.config.experimental.set_memory_growth(dev, True)
    
    train()