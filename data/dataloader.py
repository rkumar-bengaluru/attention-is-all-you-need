import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
from tensorflow.data import Dataset
import re
import string
from string import digits


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
    
    def create_encorp_datasets(self):
        hds = tfds.load('huggingface:hind_encorp')
        hds_train = hds['train']
        train_split = 0.8
        val_split = 0.2
        train_size = int(train_split * len(hds_train))
        val_size = int(val_split * len(hds_train))

        train_encorp_samples, val_encorp_samples = hds_train.take(train_size), hds_train.skip(train_size).take(val_size)
        train_encorp_ds = train_encorp_samples.shuffle(self.buffer_size).batch(self.batch_size).map(
            self.prepare_encorp_batch, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        val_encorp_ds = val_encorp_samples.shuffle(self.buffer_size).batch(self.batch_size).map(
            self.prepare_encorp_batch, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        return train_encorp_ds, val_encorp_ds
    
    def create_en_in_dataset(self, data_dir='./data/'):
        df = pd.read_csv(data_dir + 'Hindi_English_Truncated_Corpus.csv')
        df.dropna(axis = 0, inplace = True)
        self.clean_text(df)
        train_split = 0.8
        val_split = 0.2
        train_size = int(train_split * len(df))
        val_size = int(val_split * len(df))


        dataset = Dataset.from_tensor_slices((df["english_sentence"].values, df["hindi_sentence"].values))
        train_encorp_samples, val_encorp_samples = dataset.take(train_size), dataset.skip(train_size).take(val_size)
        train_encorp_ds = train_encorp_samples.shuffle(self.buffer_size).batch(self.batch_size).map(
                lambda en, hi: self.prepare_en_in_batch((en,hi)), tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        val_encorp_ds = val_encorp_samples.shuffle(self.buffer_size).batch(self.batch_size).map(
                lambda en, hi: self.prepare_en_in_batch((en,hi)), tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        return train_encorp_ds, val_encorp_ds
    
    def clean_text(self, df):
        df=df[~pd.isnull(df['english_sentence'])]
        # Lowercase all characters
        df['english_sentence']=df['english_sentence'].apply(lambda x: x.lower())
        df['hindi_sentence']=df['hindi_sentence'].apply(lambda x: x.lower())
        # Remove quotes
        df['english_sentence']=df['english_sentence'].apply(lambda x: re.sub("'", '', x))
        df['hindi_sentence']=df['hindi_sentence'].apply(lambda x: re.sub("'", '', x))
        exclude = set(string.punctuation) # Set of all special characters
        # Remove all the special characters
        df['english_sentence']=df['english_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
        df['hindi_sentence']=df['hindi_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
        # Remove all numbers from text
        remove_digits = str.maketrans('', '', digits)
        df['english_sentence']=df['english_sentence'].apply(lambda x: x.translate(remove_digits))
        df['hindi_sentence']=df['hindi_sentence'].apply(lambda x: x.translate(remove_digits))

        df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))
        df.drop_duplicates(inplace=True)

        # Remove extra spaces
        df['english_sentence']=df['english_sentence'].apply(lambda x: x.strip())
        df['hindi_sentence']=df['hindi_sentence'].apply(lambda x: x.strip())
        df['english_sentence']=df['english_sentence'].apply(lambda x: re.sub(" +", " ", x))
        df['hindi_sentence']=df['hindi_sentence'].apply(lambda x: re.sub(" +", " ", x))
    
    def prepare_en_in_batch(self, inputs):
        e_line ,h_line = inputs
        en = self.src_tokenizer.tokenize(e_line)
        en = en[:, :self.max_tokens]
        en = en.to_tensor()

        hi = self.target_tokenizer.tokenize(h_line)
        hi = hi[:, : (self.max_tokens + 1)]
        hi_inputs = hi[:, :-1].to_tensor()
        hi_labels = hi[:, 1:].to_tensor()

        return (en, hi_inputs), hi_labels
        
    
    def prepare_encorp_batch(self, k):
        e_line = k['translation']['en']
        h_line = k['translation']['hi']

        en = self.src_tokenizer.tokenize(e_line)
        en = en[:, :self.max_tokens]
        en = en.to_tensor()

        hi = self.target_tokenizer.tokenize(h_line)
        hi = hi[:, : (self.max_tokens + 1)]
        hi_inputs = hi[:, :-1].to_tensor()
        hi_labels = hi[:, 1:].to_tensor()

        return (en, hi_inputs), hi_labels
