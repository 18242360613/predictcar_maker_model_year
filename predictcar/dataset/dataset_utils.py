import tensorflow as tf
import os

LABELS_FILENAME ="lables.txt"

def int64_Features(values):
    """
    Reture TF-Features of int64s
    :param values: Scalar or a list of values
    :return:TF-featues
    """
    if not isinstance(values,(tuple,list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value = values))

def byte_Features(values):
    """
    Return TF-Features of bytes
    :param values: A Strings
    :return: tf-features
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value = [values]))

def float_Features(values):
    """
    Reture TF-Feature of float
    :param values: A float
    :return: tf-features
    """
    return tf.train.Feature(float_list = tf.train.FloatList(value=values))

def image_to_example(image_data,height,width,lable,format,filename):
    """
    Convert image to example
    :param image_data: input image
    :param height: image height
    :param width: image width
    :param lable:  num of image lable
    :param filename:
    :return: tf-example
    """
    return tf.train.Example(features = tf.train.Features(feature={
     "image/encoded":byte_Features(image_data),
     "image/height":int64_Features(height),
     "image/width":int64_Features(width),
     "image/label":int64_Features(lable),
     'image/format':byte_Features(format),
     "image/filename":byte_Features(filename)
    }))

def write_label_file(labels_to_class_names,datasetdir,filename = LABELS_FILENAME):
    """Writes a file with the list of class names.

    Args:
      labels_to_class_names: A map of (integer) labels to class names.
      dataset_dir: The directory in which the labels file should be written.
      filename: The filename where the class names are written.
    """
    labels_filename = os.path.join(datasetdir,filename)
    file = open(labels_filename,'w')
    for lable in labels_to_class_names:
        class_name = labels_to_class_names[lable]
        file.write("%d:%s\n"%(lable,class_name))
    file.close()

def has_labels(dataset_dir,filename = LABELS_FILENAME):
    """Specifies whether or not the dataset directory contains a label map file.

    Args:
      dataset_dir: The directory in which the labels file is found.
      filename: The filename where the class names are written.

    Returns:
      `True` if the labels file exists and `False` otherwise.
    """
    return tf.gfile.Exists(os.path.join(dataset_dir,filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
  """Reads the labels file and returns a mapping from ID to class name.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names
