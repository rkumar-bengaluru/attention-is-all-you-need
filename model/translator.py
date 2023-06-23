import tensorflow as tf


class TransformerTranslator(tf.Module):

    def __int__(self, tokenizers, transformer):
        super().__init__()

        self.en_tokenizer, self.pt_tokenizer = tokenizers
        self.transformer = transformer

    def __call__(self, sentence, max_length=128):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]
        sentence = self.pt_tokenizer.tokenize(sentence).to_tensor()
        encoder_input = sentence

        start_end = self.en_tokenizer.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)
            predictions = predictions[:, -1, :]
            predicted_id = tf.argmax(predictions, axis=-1)
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:
                break
        output = tf.transpose(output_array.stack())
        text = self.en_tokenizer.detokenize(output)[0]
        tokens = self.en_tokenizer.lookup(output)[0]
        self.transformer([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return text, tokens, attention_weights
