import math
import os
import random
import sys
import tensorflow as tf
import tools
from dataset import dataset_utils

_NUM_TRAIN = 61240
_NUM_VALIDATION = 10000
_NUM_TEST = 20000
_RANDOM_SEED = 0
_NUM_SHARDS = 3

class ImageReader(object):

    def __init__(self):
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data,channels=3)

    def read_image_dims(self,sess,image_data):
        image = self.decode_jpeg(sess,image_data)
        return image.shape[0],image.shape[1]

    def decode_jpeg(self,sess,image_data):
        image = sess.run(self._decode_jpeg,feed_dict={self._decode_jpeg_data:image_data})
        assert len(image.shape) ==3
        assert image.shape[2] ==3
        return image


def _get_filenames_and_classes(dataset_dir):
    """
    Returns a list of filenames and inferred class names.

    :param dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

    :return: A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
    """
    class_names = tools.get_all_label(dataset_dir)
    photo_filenames = tools.getPicslists(dataset_dir)
    return photo_filenames,sorted(class_names)

def _get_dataset_filename(dataset_dir,split_name,shard_id):
    """
    return TFrecord name
    :param dataset_dir: the directory TFrecord write to
    :param split_name: splitdata set name
    :param shard_id: the number of file split
    :return:
    """
    output_filename = 'Car_Maker_Model_Year_%s_%05d-of-%05d.tfrecord'%(split_name,shard_id,_NUM_SHARDS)
    return os.path.join(dataset_dir,output_filename)

def _convert_dataset(split_name,filenames,class_name_to_ids,dataset_dir):

    assert split_name in ['train', 'validation','test']

    num_per_shard = int(math.ceil(len(filenames)/float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session() as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(dataset_dir,split_name,shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id*num_per_shard
                    end_ndx = min((shard_id+1) *num_per_shard,len(filenames))
                    for i in list(range(start_ndx,end_ndx)):
                        try:
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                i + 1, len(filenames), shard_id))
                            sys.stdout.flush()

                            image_data = tf.gfile.FastGFile(filenames[i],'rb').read()
                            height,width = image_reader.read_image_dims(sess,image_data)
                            pathspe = filenames[i].split(os.sep)
                            imagename,class_name = pathspe[-1],pathspe[-4]+"_"+pathspe[-3]+"_"+pathspe[-2]
                            class_id = class_name_to_ids[class_name]
                            example = dataset_utils.image_to_example(image_data,height,width,class_id,'jpg'.encode(),imagename.encode())
                            tfrecord_writer.write(example.SerializeToString())
                        except:
                            print(filenames[i])
    sys.stdout.write('\n')
    sys.stdout.flush()

def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation','test']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True

def run(dataset_dir,tfrecord_dir):

    if tf.gfile.Exists(tfrecord_dir):
        tf.gfile.MakeDirs(tfrecord_dir)

    if _dataset_exists(tfrecord_dir):
        print('tfrecord files already exist. Exiting without re-creating them.')
        return

    photo_filenames,class_name = _get_filenames_and_classes(dataset_dir)
    class_name_to_ids = dict(zip(class_name,range(len(class_name))))

    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[:_NUM_TRAIN]
    validation_filenames = photo_filenames[_NUM_TRAIN:_NUM_TRAIN+_NUM_VALIDATION]
    test_filenmes = photo_filenames[_NUM_TRAIN+_NUM_VALIDATION:-1]

    _convert_dataset('train',training_filenames,class_name_to_ids,tfrecord_dir)
    _convert_dataset('validation',validation_filenames,class_name_to_ids,tfrecord_dir)
    _convert_dataset('test',test_filenmes,class_name_to_ids,tfrecord_dir)

    labels_to_class_names = dict(zip(range(len(class_name)), class_name))
    dataset_utils.write_label_file(labels_to_class_names, tfrecord_dir)

    print('\nFinished converting the afew dataset!')

if __name__ == '__main__':
    run(r'C:\Users\caopan.58GANJI-CORP\data\pic_after_filer','..\datarecord')
    # run('/home/cp/data/group_based/picture/face_filter/validation','/home/cp/data/group_based/picture/tfrecord')
    # run('/home/cp/data/group_based/picture/face_filter/train','/home/cp/data/group_based/picture/tfrecord')