import os
import tensorflow as tf
from dataset import dataset_utils

slim = tf.contrib.slim
_FILE_PATTERN = 'Car_%s_*.tfrecord'

SPLITS_TO_SIZES={'train':6998,'validation':1000,'test':2000}
_NUM_CLASSES = 10

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of fixed size.',
    'label': 'A single integer between 0 and 9',
    'parfolder': 'picture name',
}

def get_split(split_name,dataset_dir,file_pattern=None,reader=None):

    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not in recognized.'%split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN

    file_pattern = os.path.join(dataset_dir,file_pattern % split_name)

    if reader is None:
        reader = tf.TFRecordReader

    key_to_features = {
        'image/encoded':tf.FixedLenFeature((),tf.string,default_value=''),
        'image/label':tf.FixedLenFeature([],tf.int64,default_value=tf.zeros([],dtype=tf.int64)),
        'image/filename':tf.FixedLenFeature((),tf.string,default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg')
    }

    items_to_handlers = {
        'image':slim.tfexample_decoder.Image('image/encoded'),
        'label':slim.tfexample_decoder.Tensor('image/label'),
        'filename':slim.tfexample_decoder.Tensor('image/filename')
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(key_to_features,items_to_handlers)

    labels_to_name = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_name = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_name
    )
