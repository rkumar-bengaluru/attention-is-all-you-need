from abc import ABC

import tensorflow as tf


class TransformerScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_speed=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_speed = warmup_speed

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.sqrt(step)
        arg2 = step * (self.warmup_speed ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        config = {
            'd_model':self.d_model,
            'warmup_speed':self.warmup_speed

        }
        base_config = super(TransformerScheduler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

