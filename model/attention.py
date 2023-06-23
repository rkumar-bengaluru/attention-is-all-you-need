import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):

    def __int(self, **kwargs):
        super().__init__()
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):

    def __init__(self):
        super().__init__()
        self.last_attn_scores = None

    def call(self, inputs, x, context, *args, **kwargs):
        attn_output, attn_scores = self.multi_head_attention(query=x,
                                                             key=context,
                                                             value=context,
                                                             return_attention_scores=True)
        self.last_attn_scores = attn_scores
        x = self.add(x, attn_output)
        x = self.layer_norm(x)

        return x


class GlobalSelfAttention(BaseAttention):

    def call(self, x, *args, **kwargs):
        attn_output = self.multi_head_attention(query=x,
                                                value=x,
                                                key=x)
        x = self.add([x, attn_output])
        x = self.layer_norm(x)

        return x


class CasualSelfAttention(BaseAttention):

    def call(self, x, *args, **kwargs):
        attn_output = self.multi_head_attention(query=x,
                                                value=x,
                                                key=x,
                                                use_casual_mask=True)
        x = self.add([x, attn_output])
        x = self.layer_norm(x)

        return x
