import tensorflow_datasets as tfds
import tensorflow as tf


class DataLoader:

    def __init__(self, en_tokenizer, pt_tokenizer, max_tokens=128, buffer_size=20000,
                 batch_size=64):
        self.max_tokens = max_tokens
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.en_tokenizer = en_tokenizer
        self.pt_tokenizer = pt_tokenizer

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
        pt = self.pt_tokenizer.tokenize(pt)
        pt = pt[:, :self.max_tokens]
        pt = pt.to_tensor()

        en = self.en_tokenizer.tokenize(en)
        en = en[:, : (self.max_tokens + 1)]
        en_inputs = en[:, :-1].to_tensor()
        en_labels = en[:, 1:].to_tensor()

        return (pt, en_inputs), en_labels
