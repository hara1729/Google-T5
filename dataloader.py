import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import tensorflow as tf
from datasets import load_dataset, Dataset
from transformers import T5TokenizerFast
from wikiretriever import WikipediaRetriever


class WikiQARAGLoader:
    """
    Retrieval-augmented dataloader for the Microsoft WikiQA benchmark.

    Usage
    -----
    loader = WikiQARAGLoader(seq_len=128, batch_size=16,
                             num_sentences=3, cache_dir="rag_cache")
    train_ds = loader.load_split("train")
    val_ds   = loader.load_split("validation")
    """

    def __init__(
        self,
        *,
        seq_len: int = 128,
        batch_size: int = 16,
        num_sentences: int = 3,
        tokenizer_name: str = "t5-base",
        cache_dir: str | Path = "rag_cache",
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.retriever = WikipediaRetriever(num_sentences = num_sentences)
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_name)
        self.pad_id = self.tokenizer.pad_token_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents = True, exist_ok = True)
        self.seed = seed

    def load_split(self, split: Literal["train", "validation", "test"]) -> tf.data.Dataset:
        """
        Return a `tf.data.Dataset` yielding:
            (encoder_tokens, decoder_tokens), labels
        with padding masked out by the custom loss in `nn.py`.
        """
        hf_ds = self._load_and_prepare_split(split)
        tf_ds = self._to_tf_dataset(hf_ds)
        return tf_ds

    def _load_and_prepare_split(self, split: str) -> Dataset:
        """Download WikiQA and keep exactly one correct answer per question."""
        ds = load_dataset("microsoft/wiki_qa", split = split)

        def mark_keep(batch):
            pos = [i for i, lbl in enumerate(batch["label"]) if lbl == 1]
            keep = [i == pos[0] for i in range(len(batch["label"]))] if pos else [False] * len(batch["label"])
            return {"keep": keep}

        ds = ds.map(mark_keep, batched = True, batch_size = None).filter(lambda keep: keep)
        DROP_COLS = ["keep", "label", "question_id", "document_title"]
        present = [c for c in DROP_COLS if c in ds.column_names]
        ds = ds.remove_columns(present)

        cache_file = self.cache_dir / f"{split}_context.json"
        if cache_file.exists():
            with cache_file.open() as fp:
                ctx_cache = json.load(fp)
        else:
            ctx_cache = {}

        def add_context(ex):
            q = ex["question"]
            if q not in ctx_cache:
                ctx_cache[q] = self.retriever.fetch_context(q)
            ex["context"] = ctx_cache[q]
            return ex

        ds = ds.map(add_context)
        with cache_file.open("w") as fp:
            json.dump(ctx_cache, fp)
        return ds

    def _tokenise(self, question: str, context: str, answer: str) -> dict:
        prompt = f"question: {question}  context: {context}"
        enc = self.tokenizer(
                                prompt,
                                max_length = self.seq_len,
                                padding = "max_length",
                                truncation = True,
                            ).input_ids
        dec = self.tokenizer(
                                answer,
                                max_length = self.seq_len,
                                padding = "max_length",
                                truncation = True,
                            ).input_ids
        # shift decoder input right: prepend pad token, drop last answer token
        dec_input = [self.pad_id] + dec[:-1]

        return {
                    "encoder_tokens": enc,
                    "decoder_tokens": dec_input,
                    "labels"        : dec,
                }

    def _to_tf_dataset(self, hf_ds: Dataset) -> tf.data.Dataset:
        '''Convert Hugginface Dataset --> shuffled/batched `tf.data.Dataset.`'''
        colnames = hf_ds.column_names

        def encode_row(ex):
            return self._tokenise(ex["question"], ex["context"], ex["answer"])

        encoded = hf_ds.map(encode_row, remove_columns = colnames)
        encoded = encoded.with_format("tensorflow")

        # generator yields a dict of three int64 arrays
        tf_ds = tf.data.Dataset.from_generator(
                                                    lambda: encoded,
                                                    output_signature = {
                                                                            "encoder_tokens": tf.TensorSpec([self.seq_len], tf.int64),
                                                                            "decoder_tokens": tf.TensorSpec([self.seq_len], tf.int64),
                                                                            "labels":         tf.TensorSpec([self.seq_len], tf.int64),
                                                                        },
                                              )
        # split into (inputs, labels) for model.fit
        tf_ds = tf_ds.map(
                            lambda d: (
                                            (
                                                tf.cast(d["encoder_tokens"], tf.int32),
                                                tf.cast(d["decoder_tokens"], tf.int32)
                                            ),
                                        tf.cast(d["labels"], tf.int32)
                                      ),
                            num_parallel_calls = tf.data.AUTOTUNE,
                         )

        return tf_ds.batch(self.batch_size)
    


class SquadRAGLoader:
    """
    Retrieval-augmented style loader for the SQuAD dataset.

    Usage
    -----
    loader = SquadRAGLoader(seq_len=128, batch_size=16, tokenizer_name="t5-base")
    train_ds = loader.load_split("train")
    val_ds   = loader.load_split("validation")
    """

    def __init__(
                    self,
                    *,
                    seq_len: int = 128,
                    batch_size: int = 16,
                    tokenizer_name: str = "t5-base",
                    seed: int = 42,
                ):
        
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_name)
        self.pad_id = self.tokenizer.pad_token_id
        self.seed = seed

    def load_split(self, split: Literal["train", "validation"]) -> tf.data.Dataset:
        """
        Return a `tf.data.Dataset` yielding:
            (encoder_tokens, decoder_tokens), labels
        in the same format as WikiQARAGLoader.
        """
        hf_ds = load_dataset("rajpurkar/squad", split = split)
        # 1) extract one answer per example
        hf_ds = hf_ds.map(self._extract_answer, remove_columns = hf_ds.column_names)
        # 2) tokenize and convert to tf.data
        return self._to_tf_dataset(hf_ds)

    def _extract_answer(self, ex: dict) -> dict:
        # take the first answer if available
        texts = ex["answers"]["text"]
        ans = texts[0] if texts else ""
        return {
                    "question": ex["question"],
                    "context": ex["context"],
                    "answer": ans,
               }

    def _tokenise(self, question: str, context: str, answer: str) -> dict:
        # identical to WikiQARAGLoader._tokenise
        prompt = f"question: {question}  context: {context}"
        enc = self.tokenizer(
                                prompt,
                                max_length = self.seq_len,
                                padding = "max_length",
                                truncation = True,
                            ).input_ids
        dec = self.tokenizer(
                                answer,
                                max_length = self.seq_len,
                                padding = "max_length",
                                truncation = True,
                            ).input_ids
        
        # shift decoder input right
        dec_input = [self.pad_id] + dec[:-1]

        return {
                    "encoder_tokens": enc,
                    "decoder_tokens": dec_input,
                    "labels": dec,
               }

    def _to_tf_dataset(self, hf_ds: Dataset) -> tf.data.Dataset:
        """
        Convert the HuggingFace `Dataset` into a batched `tf.data.Dataset`
        yielding ((enc, dec_in), labels).
        """
        def encode_row(ex):
            return self._tokenise(ex["question"], ex["context"], ex["answer"])

        encoded = hf_ds.map(encode_row)
        encoded = encoded.map(
                                lambda ex: {
                                                "encoder_tokens": ex["encoder_tokens"],
                                                "decoder_tokens": ex["decoder_tokens"],
                                                "labels":         ex["labels"],
                                           },
                                remove_columns = ["question", "context", "answer"]
                             )
        encoded = encoded.with_format("tensorflow")

        # build from generator
        tf_ds = tf.data.Dataset.from_generator(
            lambda: encoded,
            output_signature = {
                                    "encoder_tokens": tf.TensorSpec([self.seq_len], tf.int64),
                                    "decoder_tokens": tf.TensorSpec([self.seq_len], tf.int64),
                                    "labels":         tf.TensorSpec([self.seq_len], tf.int64),
                               },
        )

        # split into ((enc, dec_in), labels)
        tf_ds = tf_ds.map(
                            lambda d: (
                                        (
                                            tf.cast(d["encoder_tokens"], tf.int32),
                                            tf.cast(d["decoder_tokens"], tf.int32),
                                        ),
                                        tf.cast(d["labels"], tf.int32),
                                      ),
                            num_parallel_calls = tf.data.AUTOTUNE,
                        )

        return tf_ds.batch(self.batch_size)
    
