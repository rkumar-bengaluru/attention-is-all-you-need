import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import tensorflow_datasets as tfds
import tensorflow_text as text

bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
bert_vocab_args = dict(vocab_size=8000,
                       reserved_tokens=reserved_tokens,
                       bert_tokenizer_params=bert_tokenizer_params,
                       learn_params={})


def load_eng_to_pt_dataset(datasetname):
    samples, _ = tfds.load(datasetname,
                           with_info=True,
                           as_supervised=True)
    return samples


def generate_ted_dataset_vocab():
    eng_pt = load_eng_to_pt_dataset('ted_hrlr_translate/pt_to_en')
    train_samples, val_samples = eng_pt['train'], eng_pt['validation']
    train_en = train_samples.map(lambda pt, en: en)
    train_pt = train_samples.map(lambda pt, en: pt)

    en_vocab = bert_vocab.bert_vocab_from_dataset(train_en.batch(1000).prefetch(2), **bert_vocab_args)
    write_vocab_file('en_ted_vocab.txt', en_vocab)
    print(f'Done with english token size {len(en_vocab)}')

    pt_vocab = bert_vocab.bert_vocab_from_dataset(train_pt.batch(1000).prefetch(2), **bert_vocab_args)
    write_vocab_file('pt_ted_vocab.txt', pt_vocab)
    print(f'Done with portuguese token size {len(pt_vocab)}')


def split_dataset(ds, train_size=0.8, val_size=0.2):
    train_len = int(len(ds) * train_size)
    val_len = int(len(ds) * val_size)
    train_all_ds = ds.take(train_len)
    val_all_ds = ds.skip(train_len).take(val_len)
    train_hindi_ds = train_all_ds.map(lambda sample: sample['translation']['hi'])
    train_en_ds = train_all_ds.map(lambda sample: sample['translation']['en'])
    val_hindi_ds = val_all_ds.map(lambda sample: sample['translation']['hi'])
    val_en_ds = val_all_ds.map(lambda sample: sample['translation']['en'])
    return train_hindi_ds, train_en_ds, val_hindi_ds, val_en_ds


def gen_encorp_dataset_vocab():
    hds = tfds.load('huggingface:hind_encorp')
    training_data = hds['train']
    train_hindi_ds, train_en_ds, val_hindi_ds, val_en_ds = split_dataset(training_data)
    hi_vocab = bert_vocab.bert_vocab_from_dataset(train_hindi_ds.batch(1000).prefetch(2), **bert_vocab_args)
    write_vocab_file('hindi_vocab_encorp.txt', hi_vocab)
    print(f'Done with hindi token with size {len(hi_vocab)}')
    en_vocab = bert_vocab.bert_vocab_from_dataset(train_en_ds.batch(1000).prefetch(2), **bert_vocab_args)
    write_vocab_file('en_vocab_encorp.txt', en_vocab)
    print(f'Done with english token with size {len(en_vocab)}')


def write_vocab_file(filepath, vocab):
    with open(filepath, 'w', encoding="utf-8") as f:
        for token in vocab:
            print(token, file=f)


def main():
    print(f'tensorflow version {tf.__version__}')
    print(f'tensorflow-text version {text.__version__}')
    print(f'tensorflow-datasets version {tfds.__version__}')
    gen_encorp_dataset_vocab()
    generate_ted_dataset_vocab()


if __name__ == "__main__":
    main()
