import os
from tensorflow.python import pywrap_tensorflow
checkpoint_path = "/home/cp/PycharmProjects/group_emotion/model/vgg_16.ckpt"
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key).shape) # Remove this is you want to print only variable names

# vgg_16/fc8/biases,vgg_16/fc8/weights