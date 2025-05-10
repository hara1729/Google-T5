import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import Optional, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, ops


class LayerNorm(layers.Layer):
    def __init__(self, hidden_size: int, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.scale = self.add_weight(
                                        name = "scale",
                                        shape = (hidden_size,),
                                        initializer = "ones",
                                        trainable = True,
                                    )

    def call(self, x):
        mean = ops.mean(x, axis = -1, keepdims = True)
        var = ops.mean(ops.square(x - mean), axis = -1, keepdims = True)
        return self.scale * (x - mean) * ops.rsqrt(var + self.epsilon)


class RelativePositionBias(layers.Layer):
    def __init__(
                    self,
                    num_heads: int,
                    num_buckets: int = 32,
                    max_distance: int = 128,
                    bidirectional: bool = True,
                    **kwargs,
                ):
        
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.embed = layers.Embedding(
                                        input_dim = num_buckets,
                                        output_dim = num_heads,
                                        name = "rpb_embed",
                                     )

    def call(self, q_len: tf.Tensor, k_len: tf.Tensor):
        ctx = ops.arange(q_len)[:, None]
        mem = ops.arange(k_len)[None, :]
        rel = mem - ctx
        buckets = self._bucket(rel)
        bias = self.embed(buckets)
        return ops.transpose(bias, axes = [2, 0, 1])

    def _bucket(self, rel):
        n = -rel
        if self.bidirectional:
            half = self.num_buckets // 2
            res = ops.cast(n < 0, tf.float32) * half
            n = ops.abs(n)
        else:
            half = self.num_buckets
            res = ops.zeros_like(n, tf.float32)
            n = ops.maximum(n, 0)

        max_exact = half // 2
        is_small = n < max_exact
        large_val = max_exact + ops.cast(
                                            ops.log(ops.cast(n, tf.float32) / max_exact + 1e-6)
                                            / ops.log(self.max_distance / max_exact)
                                            * (half - max_exact),
                                            tf.float32,
                                       )
        large_val = ops.minimum(large_val, half - 1)
        return res + ops.where(is_small, n, large_val)


class ScaledEmbedding(layers.Layer):
    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.embed = layers.Embedding(vocab_size, d_model)

    def call(self, x):
        return self.embed(x)


class SplitHeads(keras.layers.Layer):
    def __init__(self, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads

    def call(self, x):
        head_dim = tf.shape(x)[-1] // self.num_heads
        b, t = ops.shape(x)[0], ops.shape(x)[1]
        x = ops.reshape(x, newshape = (b, t, self.num_heads, head_dim))
        return ops.transpose(x, axes = (0, 2, 1, 3))

class MergeHeads(layers.Layer):
    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def call(self, x):
        x = ops.transpose(x, axes = (0, 2, 1, 3))
        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        return ops.reshape(x, newshape = (b, t, self.d_model))

    def compute_output_shape(self, input_shape):
        batch  = input_shape[0]
        length = input_shape[2]
        return (batch, length, self.d_model)






def _attention(q, k, v, mask, head_dim, dropout, rpb):
    k_t    = ops.transpose(k, axes=[0, 1, 3, 2])          # (B,H,D,T)
    scores = ops.matmul(q, k_t) / ops.sqrt(ops.cast(head_dim, q.dtype))

    # ------------------- relative position bias -------------------
    if rpb is not None:
        q_len = layers.Lambda(lambda q: tf.shape(q)[-2])(q)
        k_len = layers.Lambda(lambda k: tf.shape(k)[-2])(k)

        bias  = rpb(q_len, k_len)
        bias  = ops.expand_dims(bias, 0)
        scores = scores + bias
    # --------------------------------------------------------------

    if mask is not None:
        scores += (1.0 - ops.cast(mask, scores.dtype)) * -1e9

    attn = ops.softmax(scores, axis = -1)
    if dropout:
        attn = layers.Dropout(dropout)(attn)          
    
    return ops.matmul(attn, v)                            # (B,H,Q,D)





class MultiHeadAttention(keras.Model):
    def __init__(
                    self,
                    d_model: int,
                    num_heads: int,
                    dropout: float = 0.0,
                    use_relative_position_bias: bool = False,
                    name: str = "mha",
                    **kwargs,
                ):
            
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_relative_position_bias = use_relative_position_bias
        
        head_dim = d_model // num_heads

        q_in  = keras.Input(shape = (None, d_model), name = f"{name}_q")
        k_in  = keras.Input(shape = (None, d_model), name = f"{name}_k")
        v_in  = keras.Input(shape = (None, d_model), name = f"{name}_v")
        m_in  = keras.Input(shape = (None,None, None),  name = f"{name}_mask", dtype = "float32")

        q_proj = layers.Dense(num_heads * head_dim, use_bias = False, name = f"{name}_q_proj")
        k_proj = layers.Dense(num_heads * head_dim, use_bias = False, name = f"{name}_k_proj")
        v_proj = layers.Dense(num_heads * head_dim, use_bias = False, name = f"{name}_v_proj")
        o_proj = layers.Dense(d_model, use_bias = False, name = f"{name}_o_proj")

        rpb = RelativePositionBias(num_heads = num_heads, name = f"{name}_rpb") if use_relative_position_bias else None

        q = SplitHeads(num_heads)(q_proj(q_in))
        k = SplitHeads(num_heads)(k_proj(k_in))
        v = SplitHeads(num_heads)(v_proj(v_in))

        out = _attention(
                            q = q,
                            k = k,
                            v = v,
                            mask = m_in,
                            head_dim = head_dim,
                            dropout = dropout,
                            rpb = rpb,
                        )
        out = MergeHeads(d_model)(out)
        out = o_proj(out)

        super().__init__(inputs = [q_in, k_in, v_in, m_in], outputs = out, name = name, **kwargs)
    
    def compute_output_shape(self, input_spec, **_):
        batch  = input_spec[0].shape[0]      # same as q‑input
        length = input_spec[0].shape[1]
        return (batch, length, self.d_model)


class GroupedQueryAttention(keras.Model):
    def __init__(
                    self,
                    d_model: int,
                    num_heads: int,
                    num_query_groups: int,
                    dropout: float = 0.0,
                    use_relative_position_bias: bool = False,
                    name: str = "gqa",
                    **kwargs,
                ):
       
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_query_groups = num_query_groups
        self.dropout = dropout
        self.use_relative_position_bias = use_relative_position_bias
        
        if num_heads % num_query_groups != 0:
            raise ValueError("num_heads must be divisible by num_query_groups")
        head_dim = d_model // num_heads
        heads_per_group = num_heads // num_query_groups

        q_in = keras.Input(shape = (None, d_model), name = f"{name}_q")
        k_in = keras.Input(shape = (None, d_model), name = f"{name}_k")
        v_in = keras.Input(shape = (None, d_model), name = f"{name}_v")
        m_in = keras.Input(shape = (None, None, None),  name = f"{name}_mask", dtype = "float32")

        q_proj = layers.Dense(num_heads * head_dim,          use_bias = False, name = f"{name}_q_proj")
        k_proj = layers.Dense(num_query_groups * head_dim,   use_bias = False, name = f"{name}_k_proj")
        v_proj = layers.Dense(num_query_groups * head_dim,   use_bias = False, name = f"{name}_v_proj")
        o_proj = layers.Dense(d_model,                       use_bias = False, name = f"{name}_o_proj")

        rpb = RelativePositionBias(num_heads = num_heads, name = f"{name}_rpb") \
            if use_relative_position_bias else None

        q = SplitHeads(num_heads)(q_proj(q_in))
        k = SplitHeads(num_query_groups)(k_proj(k_in))
        v = SplitHeads(num_query_groups)(v_proj(v_in))

        k = ops.repeat(k, repeats = heads_per_group, axis = 1)
        v = ops.repeat(v, repeats = heads_per_group, axis = 1)

        out = _attention(
                            q = q,
                            k = k,
                            v = v,
                            mask = m_in,
                            head_dim = head_dim,
                            dropout = dropout,
                            rpb = rpb,
                        )
        out = MergeHeads(d_model)(out)
        out = o_proj(out)
        
        super().__init__(inputs = [q_in, k_in, v_in, m_in], outputs = out, name = name, **kwargs)

    def compute_output_shape(self, input_shape, **_):
        batch  = input_shape[0][0]
        length = input_shape[0][1]
        return (batch, length, self.d_model)

class FeedForward(keras.Model):
    def __init__(
                    self,
                    d_model: int,
                    d_ff: int,
                    dropout: float = 0.0,
                    name: str = "ffn",
                    **kwargs,
                ):
        
        x_in = keras.Input(shape = (None, d_model), name = f"{name}_in")
        out = layers.Dense(d_ff, name = f"{name}_d1")(x_in)
        out = layers.Activation(keras.activations.get("gelu"))(out)
        out = layers.Dropout(dropout, name = f"{name}_drop")(out)
        out = layers.Dense(d_model, name = f"{name}_d2")(out)
        
        super().__init__(inputs = x_in, outputs = out, name = name, **kwargs)
    
    def compute_output_shape(self, input_shape):
        return input_shape


class EncoderBlock(keras.Model):
    def __init__(
                    self,
                    d_model: int,
                    num_heads: int,
                    d_ff: int,
                    attention_type: str,
                    num_query_groups: Optional[int],
                    use_relative_position_bias: bool,
                    attn_dropout: float,
                    ffn_dropout: float,
                    epsilon: float,
                    name: str,
                    **kwargs,
                ):
        
        self.d_model = d_model
        
        x_in   = keras.Input(shape = (None, d_model), name = f"{name}_x")
        mask   = keras.Input(shape = (1, 1, None),   name = f"{name}_mask", dtype = "float32")

        ln1 = LayerNorm(d_model, epsilon, name = f"{name}_ln1")
        ln2 = LayerNorm(d_model, epsilon, name = f"{name}_ln2")

        AttnCls = GroupedQueryAttention if attention_type == "gqa" else MultiHeadAttention
        attn = AttnCls(
                            d_model = d_model,
                            num_heads = num_heads,
                            num_query_groups = num_query_groups,
                            dropout = attn_dropout,
                            use_relative_position_bias = use_relative_position_bias,
                            name = f"{name}_attn",
                      )
        ffn  = FeedForward(
                                d_model = d_model,
                                d_ff = d_ff,
                                dropout = ffn_dropout,
                                name = f"{name}_ffn",
                          )

        h = ln1(x_in)
        h = attn([h, h, h, mask])
        x = layers.Add(name = f"{name}_add1")([x_in, h])

        h = ln2(x)
        h = ffn(h)
        out = layers.Add(name = f"{name}_add2")([x, h])
        
        super().__init__(inputs = [x_in, mask], outputs = out, name = name, **kwargs)
  
    def compute_output_shape(self, input_shape):
        # input_shape = ([B,T,d_model], [B,1,1,T])
        batch  = input_shape[0][0]
        length = input_shape[0][1]
        return (batch, length, self.d_model)


class CausalMask(layers.Layer):
    '''
    Return a (1,1,T,T) lower-triangular mask where T = tf.shape(x)[1]
    '''
    def call(self, x):
        seq_len = tf.shape(x)[1]
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=x.dtype), -1, 0)
        return mask[None, None]



class DecoderBlock(keras.Model):
    def __init__(
                    self,
                    d_model: int,
                    num_heads: int,
                    d_ff: int,
                    attention_type: str,
                    num_query_groups: Optional[int],
                    use_relative_position_bias: bool,
                    attn_dropout: float,
                    ffn_dropout: float,
                    epsilon: float,
                    name: str,
                    **kwargs,
                ):
                
        self.d_model = d_model
        
        x_in   = keras.Input(shape = (None, d_model), name = f"{name}_x")
        enc_in = keras.Input(shape = (None, d_model), name = f"{name}_enc")
        pmask  = keras.Input(shape = (1, 1, None),   name = f"{name}_pad", dtype = "float32")

        ln_self  = LayerNorm(d_model, epsilon, name = f"{name}_ln_self")
        ln_cross = LayerNorm(d_model, epsilon, name = f"{name}_ln_cross")
        ln_ffn   = LayerNorm(d_model, epsilon, name = f"{name}_ln_ffn")

        AttnCls = GroupedQueryAttention if attention_type == "gqa" else MultiHeadAttention
        self_attn = AttnCls(
                                d_model = d_model,
                                num_heads = num_heads,
                                num_query_groups = num_query_groups,
                                dropout = attn_dropout,
                                use_relative_position_bias = use_relative_position_bias,
                                name = f"{name}_self",
                           )
        cross_attn = AttnCls(
                                d_model = d_model,
                                num_heads = num_heads,
                                num_query_groups = num_query_groups,
                                dropout = attn_dropout,
                                use_relative_position_bias = use_relative_position_bias,
                                name = f"{name}_cross",
                            )
        ffn = FeedForward(
                            d_model = d_model,
                            d_ff = d_ff,
                            dropout = ffn_dropout,
                            name = f"{name}_ffn",
                         )

        seq_len = ops.shape(x_in)[1]
        c_mask = CausalMask(name=f"{name}_c_mask")(x_in)
        m_self = layers.Lambda(lambda t: tf.minimum(t[0], t[1]), name = f"{name}_merge_mask")([c_mask, pmask])

        # self‑attn
        h = ln_self(x_in)
        h = self_attn([h, h, h, m_self])
        x = layers.Add(name = f"{name}_add1")([x_in, h])

        # cross‑attn
        h = ln_cross(x)
        h = cross_attn([h, enc_in, enc_in, pmask])
        x = layers.Add(name = f"{name}_add2")([x, h])

        # ffn
        h = ln_ffn(x)
        h = ffn(h)
        out = layers.Add(name = f"{name}_add3")([x, h])
        
        super().__init__(inputs = [x_in, enc_in, pmask], outputs = out, name = name, **kwargs)
   
    def compute_output_shape(self, input_shape):
        # input_shape = ([B,T,d_model], [B,S,d_model], [B,1,1,T])
        batch  = input_shape[0][0]
        length = input_shape[0][1]
        return (batch, length, self.d_model)

class T5Model(keras.Model):
    def __init__(
                    self,
                    vocab_size: int,
                    seq_len: int,
                    d_model: int = 512,
                    num_heads: int = 8,
                    d_ff: int = 2048,
                    num_layers: int = 6,
                    attention_type: str = "mha",
                    num_query_groups: Optional[int] = None,
                    use_relative_position_bias: bool = True,
                    attn_dropout: float = 0.0,
                    ffn_dropout: float = 0.0,
                    layer_norm_epsilon: float = 1e-6,
                    name: str = "t5_model",
                    **kwargs,
                ):
        
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.attention_type = attention_type
        self.num_query_groups = num_query_groups
        self.use_relative_position_bias = use_relative_position_bias
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        embed = ScaledEmbedding(vocab_size, d_model, name = "shared_embed")
        pad_to_mask = lambda t: ops.cast(ops.not_equal(t, 0), tf.float32)[:, None, None, :]

        enc_blocks: List[EncoderBlock] = [
                                                EncoderBlock(
                                                                d_model = d_model,
                                                                num_heads = num_heads,
                                                                d_ff = d_ff,
                                                                attention_type = attention_type,
                                                                num_query_groups = num_query_groups,
                                                                use_relative_position_bias = use_relative_position_bias,
                                                                attn_dropout = attn_dropout,
                                                                ffn_dropout = ffn_dropout,
                                                                epsilon = layer_norm_epsilon,
                                                                name = f"enc_blk_{i}",
                                                            )
                                                for i in range(num_layers)
                                         ]
        dec_blocks: List[DecoderBlock] = [
                                                DecoderBlock(
                                                                d_model = d_model,
                                                                num_heads = num_heads,
                                                                d_ff = d_ff,
                                                                attention_type = attention_type,
                                                                num_query_groups = num_query_groups,
                                                                use_relative_position_bias = use_relative_position_bias,
                                                                attn_dropout = attn_dropout,
                                                                ffn_dropout = ffn_dropout,
                                                                epsilon = layer_norm_epsilon,
                                                                name = f"dec_blk_{i}",
                                                            )
                                                for i in range(num_layers)
                                         ]

        enc_norm = LayerNorm(d_model, layer_norm_epsilon, name = "enc_norm")
        dec_norm = LayerNorm(d_model, layer_norm_epsilon, name = "dec_norm")
        
        enc_tokens = keras.Input(shape = (seq_len,), name = "encoder_tokens")
        dec_tokens = keras.Input(shape = (seq_len,), name = "decoder_tokens")

        enc_mask = layers.Lambda(pad_to_mask, name = "enc_pad")(enc_tokens)
        dec_mask = layers.Lambda(pad_to_mask, name = "dec_pad")(dec_tokens)

        h_enc = embed(enc_tokens)
        for blk in enc_blocks:
            h_enc = blk([h_enc, enc_mask])
        h_enc = enc_norm(h_enc)

        h_dec = embed(dec_tokens)
        for blk in dec_blocks:
            h_dec = blk([h_dec, h_enc, dec_mask])
        h_dec = dec_norm(h_dec)

        # weight tying
        logits = layers.Lambda(
                                    lambda z: ops.matmul(z, ops.transpose(embed.embed.embeddings)),
                                    name = "logits",
                              )(h_dec)

       
        super().__init__(inputs = [enc_tokens, dec_tokens], outputs = logits, name = name, **kwargs)


    def compute_output_shape(self, input_shape):
        batch = input_shape[0][0]
        return (batch, self.seq_len, self.vocab_size)
    
    def get_config(self):
        config = super().get_config()
        config.update({
                            'seq_len': self.seq_len,
                            'vocab_size': self.vocab_size,
                            'd_model': self.d_model,
                            'd_ff': self.d_ff,
                            'num_layers': self.num_layers,
                            'attention_type': self.attention_type,
                            'num_query_groups': self.num_query_groups,
                            'use_relative_position_bias': self.use_relative_position_bias,
                            'attn_dropout': self.attn_dropout,
                            'ffn_dropout': self.ffn_dropout,
                            'layer_norm_epsilon': self.layer_norm_epsilon
                      })
        
        return config
    
if __name__ == "__main__":
    m = T5Model(
                vocab_size = 32128,
                seq_len = 16,
                d_model = 256,
                num_heads = 4,
                num_layers = 2,
                attention_type = "gqa",
                num_query_groups = 2,
                use_relative_position_bias = True,
               )
    
    m.summary(expand_nested = False, line_length = 120)