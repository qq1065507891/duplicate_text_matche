import tensorflow as tf

from tensorflow.keras.layers import Dropout
from tensorflow.keras import Model

import os
os.environ['TF_KERAS'] = '1'

from bert4keras.models import build_transformer_model
from bert4keras.backend import set_gelu
from bert4keras.layers import Lambda


class Bert_Q(object):
    """
    模型类
    """
    def __init__(self, config, last_activation='softmax', model_type='bert', dropout_rate=0):

        self.last_activation = last_activation
        self.model_type = model_type
        self.dropout_rate = dropout_rate
        self.config = config

        self.bert = build_transformer_model(
            config_path=config.config_path,
            checkpoint_path=config.checkpoint_path,
            model=self.model_type,
            return_keras_model=False
        )
        self.fc = tf.keras.layers.Dense(units=self.config.fc1, activation='relu', name='fc1')
        self.output = tf.keras.layers.Dense(units=self.config.num_classes,
                                            activation=self.last_activation, name='output')
        self.drop_out = Dropout(self.config.drop_out)

    def build_model(self):
        set_gelu('tanh')
        output = Lambda(lambda x: x[:, 0], name='CLS-token')(self.bert.model.output)

        if 0 < self.dropout_rate < 1:
            output = self.drop_out(output)

        fc = self.fc(output)

        if 0< self.dropout_rate < 1:
            fc = self.drop_out(fc)

        output = self.output(fc)

        model = Model(inputs=self.bert.model.input, outputs=output, name=self.config.model_name)

        return model




