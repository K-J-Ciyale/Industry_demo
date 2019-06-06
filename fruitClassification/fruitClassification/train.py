#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf

import fruitClassification as model

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('record_path', None, 'Path to training tfrecord file.')
flags.DEFINE_string('logdir', None, 'Path to log directory.')
FLAGS = flags.FLAGS
wid = 64
height = 48

def get_record_dataset(record_path,
                       reader=None, image_shape=[wid, height, 3], 
                       num_samples=15260, num_classes=10):
    """Get a tensorflow record file.
    
    Args:
        
    """
    if not reader:
        reader = tf.TFRecordReader
        
    keys_to_features = {
        'image/encoded': 
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': 
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label': 
            tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], 
                               dtype=tf.int64))}
        
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=image_shape, 
                                              #image_key='image/encoded',
                                              #format_key='image/format',
                                              channels=3),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])}
    
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
    
    labels_to_names = None
    items_to_descriptions = {
        'image': 'An image with shape image_shape.',
        'label': 'A single integer between 0 and 9.'}
    return slim.dataset.Dataset(
        data_sources=record_path,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        num_classes=num_classes,
        items_to_descriptions=items_to_descriptions,
        labels_to_names=labels_to_names)


def main(_):
    dataset = get_record_dataset('H:\\yan\\fruitClassification2\\datasets\\train.record')
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    image, label = data_provider.get(['image', 'label'])
    inputs, labels = tf.train.batch([image, label],
                                    batch_size=100,
                                    allow_smaller_final_batch=True)
    
    cls_model = model.Model(is_training=True, num_classes=10)
    preprocessed_inputs = cls_model.preprocess(inputs)
    prediction_dict = cls_model.predict(preprocessed_inputs)
    loss_dict = cls_model.loss(prediction_dict, labels)
    loss = loss_dict['loss']
    tf.summary.scalar('loss', loss)
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    classes = postprocessed_dict['classes']
    acc = tf.reduce_mean(tf.cast(tf.equal(classes, labels), 'float'))
    tf.summary.scalar('accuracy', acc)
    
    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9, use_nesterov = True)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0009)
    train_op = slim.learning.create_train_op(loss, optimizer, summarize_gradients=True)
    
    slim.learning.train(train_op=train_op, logdir='H:\\yan\\fruitClassification2\\datasets\\path_to_log_directory',
                        save_summaries_secs=20, save_interval_secs=120)
    
if __name__ == '__main__':
    tf.app.run()
'H:\\yan\\fruitClassification\\datasets\\train.record'
'H:\\yan\\fruitClassification\\datasets\\path_to_log_directory'