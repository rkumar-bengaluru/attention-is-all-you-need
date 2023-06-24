import tensorflow as tf
import tensorflow_datasets as tfds
from vocabulary.gen_vocab import reserved_tokens
from vocabulary.tokenizer import TransformerTokenizer

MAX_TOKENS = 128
BUFFER_SIZE = 20000
BATCH_SIZE = 32


def create_ted_datasets_raw():
    ted_samples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                      with_info=True,
                                      as_supervised=True)
    train_samples, val_samples = ted_samples['train'], ted_samples['validation']
    return train_samples,val_samples
    
def create_ted_datasets():
    ted_samples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                      with_info=True,
                                      as_supervised=True)
    train_samples, val_samples = ted_samples['train'], ted_samples['validation']
    train_ted_ds = train_samples.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).map(
        prepare_ted_batch, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ted_ds = val_samples.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).map(
        prepare_ted_batch, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ted_ds, val_ted_ds

def create_encorp_datasets():
    hds = tfds.load('huggingface:hind_encorp')
    hds_train = hds['train']
    train_split = 0.8
    val_split = 0.2
    train_size = int(train_split * len(hds_train))
    val_size = int(val_split * len(hds_train))

    train_encorp_samples, val_encorp_samples = hds_train.take(train_size), hds_train.skip(train_size).take(val_size)
    train_encorp_ds = train_encorp_samples.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).map(
        prepare_encorp_batch, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_encorp_ds = val_encorp_samples.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).map(
        prepare_encorp_batch, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_encorp_ds, val_encorp_ds

def prepare_ted_batch(pt, en):
    en_tokenizer, pt_tokenizer = get_ted_tokenizer()

    pt = pt_tokenizer.tokenize(pt)
    pt = pt[:, :MAX_TOKENS]
    pt = pt.to_tensor()

    en = en_tokenizer.tokenize(en)
    en = en[:, : (MAX_TOKENS + 1)]
    en_inputs = en[:, :-1].to_tensor()
    en_labels = en[:, 1:].to_tensor()

    return (pt, en_inputs), en_labels


def prepare_encorp_batch(k):
    eng_tokenizer, hi_tokenizer = get_encorp_tokenizer()
    e_line = k['translation']['en']
    h_line = k['translation']['hi']

    en = eng_tokenizer.tokenize(e_line)
    en = en[:, :MAX_TOKENS]
    en = en.to_tensor()

    hi = hi_tokenizer.tokenize(h_line)
    hi = hi[:, : (MAX_TOKENS + 1)]
    hi_inputs = hi[:, :-1].to_tensor()
    hi_labels = hi[:, 1:].to_tensor()

    return (en, hi_inputs), hi_labels
def get_ted_tokenizer():
    en_tokenizer = TransformerTokenizer('/content/attention-is-all-you-need/data/en_ted_vocab.txt', res_tokens=reserved_tokens)
    pt_tokenizer = TransformerTokenizer('/content/attention-is-all-you-need/data/pt_ted_vocab.txt', res_tokens=reserved_tokens)
    return en_tokenizer, pt_tokenizer

def get_encorp_tokenizer():
    en_tokenizer = TransformerTokenizer('/content/attention-is-all-you-need/data/en_vocab_encorp.txt', res_tokens=reserved_tokens)
    hi_tokenizer = TransformerTokenizer('/content/attention-is-all-you-need/data/hindi_vocab_encorp.txt', res_tokens=reserved_tokens)
    return en_tokenizer, hi_tokenizer
