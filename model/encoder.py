import tensorflow as tf
from model.attention import GlobalSelfAttention
from model.feed_forward import FeedForward
from model.positional_embedding import PositionalEmbedding


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(num_heads=num_heads,
                                                  key_dim=d_model,
                                                  dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)

        return x


class EncoderBlock(tf.keras.layers.Layer):

    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.positional_embedding = PositionalEmbedding(vocab_size, d_model)
        self.encoder = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.positional_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.encoder[i](x)

        return x
