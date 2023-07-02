import tensorflow as tf
from model.transformer import Transformer
from data_loader import get_ted_tokenizer
from model.schedule import TransformerScheduler
from data_loader import BATCH_SIZE


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


class TransformerTraining:

    def __init__(self,num_layers=4, d_mode=512, dff=2048,
                num_heads=8, dropout_rate=0.1, num_epochs=5,
                steps_per_epochs=0.1,
                save_freq=5):
        self.num_layers = num_layers
        self.d_model = d_mode
        self.dff = dff
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        self.steps_per_epochs = steps_per_epochs
        self.en_tokenizer, self.pt_tokenizer = get_ted_tokenizer()
        self.scheduler = TransformerScheduler(self.d_model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.scheduler,
                                                  beta_1=0.9,
                                                  beta_2=0.98,
                                                  epsilon=1e-9)
        self.transformer = Transformer(num_layers=self.num_layers,
                                       d_model=self.d_model,
                                       num_heads=self.num_heads,
                                       dff=self.dff,
                                       src_vocab_size=self.pt_tokenizer.get_vocab_size(),
                                       target_vocab_size=self.en_tokenizer.get_vocab_size(),
                                       dropout_rate=self.dropout_rate)
        self.checkpoint_path = "training/cp-{epoch:04d}.ckpt"
        # Create a callback that saves the model's weights
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, 
                                                              verbose=1, 
                                                              save_weights_only=True,
                                                              save_freq=save_freq*BATCH_SIZE)
        self.training_history = None

    def get_model(self):
        return self.transformer
    
    def get_schedule(self):
        return self.scheduler
    
    def compile(self):
        self.transformer.compile(loss=masked_loss,
                                 optimizer=self.optimizer,
                                 metrics=[masked_accuracy])

    def fit(self, train_batches, val_batches):
        self.training_history = self.transformer.fit(train_batches,
                                                     steps_per_epoch=int(self.steps_per_epochs * len(train_batches)),
                                                     epochs=self.num_epochs,
                                                     validation_data=val_batches,
                                                     validation_steps=int(self.steps_per_epochs * len(val_batches)),
                                                    callbacks=[self.cp_callback])
        return self.training_history
