import tensorflow as tf
import os
import time
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.model.InitModel import init_model, load_model
from src.utils.process import process_text
from src.utils.utils import get_time_idf, training_curve, map_example_to_dict, log


def training(config):
    """
    训练模型
    :param config:
    :return:
    """

    log.info('************数据预处理***************')
    start_time = time.time()

    train, test = process_text()

    end_time = get_time_idf(start_time)
    log.info(f'数据处理完成， 用时： {end_time}')

    log.info('************数据加载中**************')
    start_time = time.time()

    train_input_ids, train_segment_ids, train_label = train['token_ids'], train['segment_ids'], train['label']

    test_input_ids, test_segment_ids, test_label = test['token_ids'], test['segment_ids'], test['label']

    train_iter = tf.data.Dataset.from_tensor_slices((train_input_ids, train_segment_ids, train_label))\
        .map(map_example_to_dict).shuffle(50).batch(config.batch_size, drop_remainder=True)

    test_iter = tf.data.Dataset.from_tensor_slices((test_input_ids, test_segment_ids, test_label))\
        .map(map_example_to_dict).shuffle(50).batch(config.batch_size, drop_remainder=True)

    end_time = get_time_idf(start_time)

    log.info(f'数据加载完成, 用时:{end_time}, 训练数据: {len(list(train_iter))}, 验证数据： {len(list(test_iter))}')

    log.info('***********开始训练***************')

    if os.path.exists(config.model_path):
        log.info('*************已有模型， 加载模型****************')
        model = load_model(config)
    else:
        log.info('**************没有模型，初始化模型***************')
        model = init_model(config)

    calL_backs = [
        # EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.0001,
                          cooldown=0, min_lr=0),
        ModelCheckpoint(config.model_path, monitor='val_loss', verbose=1, save_best_only=True,
                        mode='max', period=1, save_weights_only=True)
    ]

    start_time = time.time()

    history = model.fit(
        train_iter,
        validation_data=test_iter,
        batch_size=config.batch_size,
        epochs=config.epochs,
        callbacks=calL_backs
    )
    end_time = get_time_idf(start_time)
    log.info(f'训练完成， 用时： {end_time}')
    training_curve(history.history['loss'], history.history['acc'],
                   history.history['val_loss'], history.history['val_acc'])

