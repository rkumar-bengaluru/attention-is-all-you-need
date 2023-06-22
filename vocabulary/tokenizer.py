import tensorflow as tf
import tensorflow_text as text
from pathlib import Path
import re


class TransformerTokenizer(tf.Module):
    """Transformer bert tokenizer wrapper

    """

    def __init__(self, vocab_path, res_tokens):
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = res_tokens
        self._vocab_path = vocab_path

        self.vocab = tf.Variable(Path(vocab_path).read_text('utf-8').splitlines())
        self.START = tf.argmax(tf.constant(res_tokens) == "[START]")
        self.END = tf.argmax(tf.constant(res_tokens) == "[END]")


    @tf.function
    def tokenize(self, strings):
        encode = self.tokenizer.tokenize(strings)
        # merge the word and word-piece axis
        encode = encode.merge_dims(-2, -1)
        encode = self.add_start_end(encode)
        return encode

    @tf.function
    def detokenize(self, strings):
        words = self.tokenizer.detokenize(strings)
        return self.clean_up(words)

    @tf.function
    def lookup(self, tokens):
        return tf.gather(self.vocab, tokens)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    def add_start_end(self, encode):
        count = encode.bounding_shape()[0]
        starts = tf.fill([count, 1], self.START)
        ends = tf.fill([count, 1], self.END)
        return tf.concat([starts, encode, ends], axis=1)

    def clean_up(self, words):
        bad_tokens = [re.escape(tok) for tok in self._reserved_tokens if tok != "[UNK]"]
        bad_tokens_re = "|".join(bad_tokens)
        bad_cells = tf.strings.regex_full_match(words, bad_tokens_re)
        result = tf.ragged.boolean_mask(words, ~bad_cells)
        result = tf.strings.reduce_join(result, separator=' ', axis=1)
        return result


