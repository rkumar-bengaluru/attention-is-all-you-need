from data_loader import create_ted_datasets, create_encorp_datasets, get_ted_tokenizer, get_encorp_tokenizer
import tensorflow_text as text
from pathlib import Path
import tensorflow as tf


def test_ted_batch():
    train_batches, val_batches = create_ted_datasets()
    en_tokenizer, pt_tokenizer = get_ted_tokenizer()

    for (pt_line, en_line), label in train_batches.take(1):
        print('pt_line\n', pt_line, type(pt_line))
        print('en_line\n', en_line, type(en_line))
        print('label\n', label, type(label))

        pt_tokens = pt_tokenizer.detokenize(pt_line.numpy())
        en_tokens = en_tokenizer.detokenize(en_line.numpy())

        print('pt line\n', pt_tokens.numpy()[0].decode('utf-8'))
        print('pt tokens\n', pt_tokenizer.lookup(pt_line.numpy()))
        print('en line\n', en_tokens.numpy()[0].decode('utf-8'))
        print('en tokens\n', en_tokenizer.lookup(en_line.numpy()))
        label_tokens = en_tokenizer.detokenize(label.numpy())
        print('en label\n', label_tokens.numpy()[0].decode('utf-8'))
        print('label tokens\n', en_tokenizer.lookup(label.numpy()))

        print(pt_line.shape)
        print(en_line.shape)
        print(label.shape)

def test_encorp_batch():
    train_batches, val_batches = create_encorp_datasets()
    en_tokenizer, hi_tokenizer = get_encorp_tokenizer()

    for (en_line, hi_line), label in train_batches.take(1):
        print('en_line\n', en_line, type(en_line))
        print('hi_line\n', hi_line, type(hi_line))
        print('label\n', label, type(label))

        pt_tokens = en_tokenizer.detokenize(en_line.numpy())
        en_tokens = hi_tokenizer.detokenize(hi_line.numpy())

        print('en_line\n', pt_tokens.numpy()[0].decode('utf-8'))
        print('en tokens\n', en_tokenizer.lookup(en_line.numpy()))
        print('hi line\n', en_tokens.numpy()[0].decode('utf-8'))
        print('hi tokens\n', hi_tokenizer.lookup(hi_line.numpy()))
        label_tokens = en_tokenizer.detokenize(label.numpy())
        print('hi label\n', label_tokens.numpy()[0].decode('utf-8'))
        print('label tokens\n', hi_tokenizer.lookup(label.numpy()))

        print(en_line.shape)
        print(hi_line.shape)
        print(label.shape)
def main():
    #test_ted_batch()
    test_encorp_batch()


if __name__ == "__main__":
    main()
