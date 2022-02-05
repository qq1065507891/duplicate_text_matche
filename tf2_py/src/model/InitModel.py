import tensorflow as tf

from src.model.bert_model import Bert_Q
from src.model.fcoal_loss import binary_focal_loss


def init_model(config):
    """
    初始化模型
    :param config:
    :return:
    """
    model = Bert_Q(config, last_activation=config.last_activation, dropout_rate=config.drop_out).build_model()

    optim = tf.keras.optimizers.Adam(config.learning_rate)
    # loss = tf.keras.losses.binary_crossentropy
    # metrics = tf.keras.metrics.binary_accuracy
    metrics = ['acc']

    model.compile(optimizer=optim, loss=binary_focal_loss, metrics=metrics)

    return model


def load_model(config):
    """
    加载模型
    :param config:
    :return:
    """
    model = init_model(config)
    model.load_weights(config.model_path)
    return model
