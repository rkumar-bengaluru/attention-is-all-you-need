import tensorflow_datasets as tfds
import tensorflow as tf


class DataLoader:

    def __init__(self, src_tokenizer, target_tokenizer, max_tokens=128, buffer_size=20000,
                 batch_size=64):
        self.max_tokens = max_tokens
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_tokenizer = target_tokenizer
        self.src_tokenizer = src_tokenizer

    def create_ted_datasets(self):
        ted_samples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                          with_info=True,
                                          as_supervised=True)
        train_samples, val_samples = ted_samples['train'], ted_samples['validation']
        train_ted_ds = train_samples.shuffle(self.buffer_size).batch(self.batch_size).map(
            self.prepare_ted_batch, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        val_ted_ds = val_samples.shuffle(self.buffer_size).batch(self.batch_size).map(
            self.prepare_ted_batch, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        return train_ted_ds, val_ted_ds

    def prepare_ted_batch(self, pt, en):
        pt = self.src_tokenizer.tokenize(pt)
        pt = pt[:, :self.max_tokens]
        pt = pt.to_tensor()

        en = self.target_tokenizer.tokenize(en)
        en = en[:, : (self.max_tokens + 1)]
        en_inputs = en[:, :-1].to_tensor()
        en_labels = en[:, 1:].to_tensor()

        return (pt, en_inputs), en_labels
    
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
    
    def prepare_encorp_batch(k):
        eng_tokenizer, hi_tokenizer = get_encorp_tokenizer()
        e_line = k['translation']['en']
        h_line = k['translation']['hi']

        en = src_tokenizer.tokenize(e_line)
        en = en[:, :MAX_TOKENS]
        en = en.to_tensor()

        hi = target_tokenizer.tokenize(h_line)
        hi = hi[:, : (MAX_TOKENS + 1)]
        hi_inputs = hi[:, :-1].to_tensor()
        hi_labels = hi[:, 1:].to_tensor()

        return (en, hi_inputs), hi_labels
