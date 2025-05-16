import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import pathlib

import tensorflow as tf
from tensorflow import keras

from dataloader import WikiQARAGLoader
from wikiretriever import WikipediaRetriever
from nn import T5Trainer

t5_kwargs = json.loads(pathlib.Path("t5_kwargs.json").read_text())

SEQ_LEN     = t5_kwargs['seq_len']
CACHE_DIR   = pathlib.Path("rag_cache")
TOKENISER   = WikiQARAGLoader(seq_len = SEQ_LEN).tokenizer
PAD_ID      = TOKENISER.pad_token_id
BOS_ID      = PAD_ID
EOS_ID      = TOKENISER.eos_token_id or 1
RETRIEVER   = WikipediaRetriever(num_sentences = 10)


WEIGHT_FILE = "./saved_models/t5-1.keras"

def _load_trained_model():
    trainer = T5Trainer(
                            pad_id = PAD_ID,
                            vocab_size = TOKENISER.vocab_size,
                            t5_kwargs = t5_kwargs
                        )
    
    t5_model = keras.models.load_model(WEIGHT_FILE)
    wrapper = lambda enc_tokens, dec_tokens, training: t5_model([enc_tokens, dec_tokens], training = training)
    
    return wrapper

_T5 = None

def _top_k_mask(logits, k):
    if k is None:
        return logits
    values, _ = tf.math.top_k(logits, k = k)
    min_values = values[:, -1, tf.newaxis]
    return tf.where(logits < min_values, tf.float32.min, logits)

def _top_p_mask(logits, p):
    if p is None:
        return logits
    sorted_logits, sorted_idx = tf.math.top_k(logits, k = tf.shape(logits)[-1])
    cumprobs = tf.math.cumsum(tf.nn.softmax(sorted_logits), axis = -1)
    mask = cumprobs > p
    mask = tf.concat([tf.zeros_like(mask[:, :1], tf.bool), mask[:, :-1]], axis = -1)
    sorted_logits = tf.where(mask, tf.float32.min, sorted_logits)
    return tf.scatter_nd(tf.expand_dims(sorted_idx, -1), sorted_logits, tf.shape(logits))

# @tf.function(jit_compile = True)
def _sample_loop(model, enc_tokens, max_len, temperature,
                 top_k, top_p, greedy):

    dec = tf.fill([1, 1], BOS_ID)          # [1, 1]
    finished = tf.constant(False)
    # tf.autograph.experimental.set_loop_options(shape_invariants=[(dec, tf.TensorShape([1, None]))])
    for _ in tf.range(max_len):
        pad_len   = SEQ_LEN - tf.shape(dec)[1]
        dec_padded = tf.pad(dec,[[0, 0], [0, pad_len]], constant_values = PAD_ID)
        all_logits = model(enc_tokens, dec_padded, training = False)
        step       = tf.shape(dec)[1] - 1
        logits     = all_logits[:, step, :]
        if greedy:
            next_id = tf.argmax(logits, axis = -1, output_type = tf.int32)
        else:
            logits = logits / temperature
            logits = _top_k_mask(logits, top_k)
            logits = _top_p_mask(logits, top_p)
            next_id = tf.random.categorical(logits, 1, dtype = tf.int32)[:, 0]
        dec = tf.concat([dec, next_id[:, None]], axis = 1)
        finished |= tf.equal(next_id, EOS_ID)
        if finished:
            break
    return dec[:, 1:]

def answer(question: str,
           *,
           seed: int | None = None,
           temperature: float = 1.0,
           top_k: int | None = None,
           top_p: float | None = None,
           beam_size: int | None = None,
           max_length: int = 64) -> str:
    """
    Generate a (possibly stochastic) answer.

    Parameters
    ----------
    question : str
    seed, temperature, top_k, top_p : usual decoding knobs
    beam_size : set for deterministic beam search (not stochastic)
    max_length : decoder cutoff

    Returns
    -------
    answer string
    """
    global _T5
    if _T5 is None:
        _T5 = _load_trained_model()

    if seed is not None:
        tf.random.set_seed(seed)

    context = RETRIEVER.fetch_context(question)
    print(f"fetched context: {context}")
    prompt  = f"question: {question}  context: {context}"
    enc_ids = TOKENISER(
                           prompt,
                           max_length = SEQ_LEN,
                           padding = "max_length",
                           truncation = True,
                       ).input_ids
    
    enc_ids = tf.convert_to_tensor(enc_ids, dtype = tf.int32)
    
    enc_tokens = enc_ids[None, :]

    if beam_size:
        # beam search
        sequences = [([BOS_ID], 0.0)]
        for _ in range(max_length):
            all_next = []
            for seq, score in sequences:
                if seq[-1] == EOS_ID:
                    all_next.append((seq, score))
                    continue
                dec = tf.constant(seq, tf.int32)[None, :]
                logits = _T5([enc_tokens, dec], training = False)[:, -1, :]
                logprobs = tf.nn.log_softmax(logits)[0].numpy()
                top_idx = logprobs.argsort()[-beam_size:][::-1]
                all_next.extend([(seq+[i], score + logprobs[i]) for i in top_idx])

            sequences = sorted(all_next, key = lambda t: t[1], reverse = True)[:beam_size]
            if all(s[-1] == EOS_ID for s,_ in sequences):
                break
        best = sequences[0][0][1:]
        return TOKENISER.decode(best, skip_special_tokens = True)

    greedy = (temperature == 0.0 and top_k is None and top_p is None)
    dec_tokens = _sample_loop(_T5, enc_tokens, max_length,
                              max(temperature, 1e-5), top_k, top_p, greedy)
    return TOKENISER.decode(dec_tokens[0].numpy(),
                            skip_special_tokens = True)
