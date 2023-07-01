from data_loader import create_ted_datasets, create_encorp_datasets, get_ted_tokenizer, get_encorp_tokenizer
import tensorflow_text as text
from pathlib import Path
import tensorflow as tf
from train import TransformerTraining
from model.transformer import Transformer
from model.testmodel import TestModel
from model.positional_embedding import PositionalEmbedding

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


def train():
    
   
    train_ted_ds, val_ted_ds = create_ted_datasets()
    training = TransformerTraining(num_epochs=20,
                                   steps_per_epochs=1)
    training.compile()
    training_history = training.fit(train_ted_ds, val_ted_ds)
    print(training_history)

def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred
    mask = label != 0
    match = match & mask
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction='none')
    loss = loss_object(label, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

    return loss
def test_model():
    train_ted_ds, val_ted_ds = create_ted_datasets()
    en_tokenizer, pt_tokenizer = get_ted_tokenizer()
    for (pt, en), en_labels in train_ted_ds.take(1):
        break


    embed_pt = PositionalEmbedding(vocab_size=pt_tokenizer.get_vocab_size(), d_model=512)
    pt_embed = embed_pt(pt)
    # print('pt_embed', pt_embed.shape)
    model = TestModel(d_model=512,
                      src_vocab_size=pt_tokenizer.get_vocab_size(),
                      target_vocab_size=en_tokenizer.get_vocab_size())
    response = model((pt, en))
    # print(response.shape)
    optimizer = tf.keras.optimizers.Adam(beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    model.compile(loss=masked_loss,
                  optimizer=optimizer,
                  metrics=[masked_accuracy])
    model.fit(train_ted_ds,
              steps_per_epoch=int(0.1* len(train_ted_ds)),
              epochs=2,
              validation_data=val_ted_ds,
              validation_steps=int(0.1*len(val_ted_ds)))

def test_transformer():
    train_ted_ds, val_ted_ds = create_ted_datasets()
    en_tokenizer, pt_tokenizer = get_ted_tokenizer()
    for (pt, en), en_labels in train_ted_ds.take(1):
        break
    transformer_model = Transformer(num_layers=1,
                                    d_model=512,
                                    num_heads=1,
                                    dff=2048,
                                    src_vocab_size=pt_tokenizer.get_vocab_size(),
                                    target_vocab_size=en_tokenizer.get_vocab_size())
    response = transformer_model((pt,en))
    print(response.shape)
def main():
    # test_ted_batch()
    # test_encorp_batch()
    # encoder = Transformer(num_layers=2,d_model=2,num_heads=2,dff=2,src_vocab_size=2,target_vocab_size=2,dropout_rate=0.1)
    # train()
    test_model()
    #test_transformer()


if __name__ == "__main__":
    main()
