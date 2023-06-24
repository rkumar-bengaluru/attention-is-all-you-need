import tensorflow as tf
from model.attention import CasualSelfAttention, CrossAttention
from model.feed_forward import FeedForward
from model.positional_embedding import PositionalEmbedding

class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.casual_self_attn = CasualSelfAttention(num_heads=num_heads,
                                                    key_dim=d_model,
                                                    dropout=dropout_rate)
        self.cross_attn = CrossAttention(num_heads=num_heads,
                                         key_dim=d_model,
                                         dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)
        self.last_attn_scores = None

    def call(self, x, context):
        x = self.casual_self_attn(x=x)
        x = self.cross_attn(x=x, context=context)
        self.last_attn_scores = self.cross_attn.last_attn_scores

        x = self.ffn(x)

        return x


class DecoderBlock(tf.keras.layers.Layer):

    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super(DecoderBlock, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.positional_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                        d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.decoder_layer = [
            DecoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(self.num_layers)]
        self.last_attn_scores = None

    def call(self, x, context):
        x = self.positional_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.decoder_layer[i](x, context)

        self.last_attn_scores = self.decoder_layer[-1].last_attn_scores
        
        return x

