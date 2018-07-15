# -*- coding:utf-8 -*-

import tensorflow as tf
import os
import numpy as np
import io
from PIL import Image


def conv(name, x, out_channel, kenerl_size=[3,3], strides=[1,1,1,1], padding = "SAME", ispretrain=True, relu=True):
    # 对卷积操作进行封装，
    '''
    :param name: 该层的名称，比如conv1,pool
    :param x: 输入的带卷积的变量
    :param out_channel:输出结果的深度，也就是卷积核心的个数
    :param kenerl_size:卷积核
    :param strides:卷积移动的步长
    :param padding:补零的方式
    :param ispretrain:该层参数是否进行预训练
    :param relu:是否使用rel激活函数
    :return:卷积的结果
    '''
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(name):
        w = tf.get_variable(name="weights",
                            shape=[kenerl_size[0], kenerl_size[1], in_channels, out_channel],
                            dtype=tf.float32,
                            trainable=ispretrain,
                            initializer=tf.contrib.layers.xavier_initializer()
                            );
        b = tf.get_variable(name = "biases",
                            shape = [out_channel],
                            trainable = ispretrain,
                            initializer=tf.constant_initializer(0.0)
                            )
        x = tf.nn.conv2d(input=x,filter=w,strides=strides,padding=padding,name='conv')
        x = tf.nn.bias_add(x,bias=b,name="biases")
        if relu:
            x = tf.nn.relu(x,name="relu")
    return x

def pool(name,x,kernel=[1,2,2,1],strides = [1,2,2,1],padding = "SAME",is_max_pool = True):
    '''
    对池化操作进行封装
    :param name: 该层的名称，比如conv1，pool
    :param x: 池化层输入的变量
    :param kernel: 池化的核心
    :param strides: 移动步长
    :param padding: 补零的方式
    :param is_max_pool: 是否是最大池化
    :return: 池化的结果
    '''

    if is_max_pool:
        x = tf.nn.max_pool(value=x,ksize=kernel,strides=strides,padding=padding,name=name)
    else:
        x = tf.nn.avg_pool(value=x,ksize=kernel,strides=strides,padding=padding,name=name)
    return x

def Fc_layer(name,x,out_node,relu = True):
    '''
    对全链接层进行封装
    :param name:
    :param x:
    :param out_node:
    :return:
    '''
    shape = x.get_shape()
    if len(shape)==4:
        size = shape[1].value*shape[2].value*shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(name):
        w = tf.get_variable(name="weights",
                            shape=[size,out_node],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="biases",
                            shape=[out_node],
                            initializer=tf.contrib.layers.xavier_initializer())

        flat_x = tf.reshape(x,shape=[-1,size])
        x = tf.nn.bias_add(tf.matmul(flat_x,w),b)
        if relu:
            x = tf.nn.relu(x)
    return x

def softmax(input, name):
    '''
    :param input:
    :param name:
    :return:
    '''
    # input_shape = map(lambda v: v.value, input.get_shape())
    # if len(input_shape) > 2:
    #     # For certain models (like NiN), the singleton spatial dimensions
    #     # need to be explicitly squeezed, since they're not broadcast-able
    #     # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
    #     if input_shape[1] == 1 and input_shape[2] == 1:
    #         input = tf.squeeze(input, squeeze_dims=[1, 2])
    #     else:
    #         raise ValueError('Rank 2 tensor input expected for softmax!')
    return tf.nn.softmax(input, name=name)

# %%
def loss(logits, labels):
    '''Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    '''
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope + '/loss', loss)
        return loss,cross_entropy


def load_with_skip(modelpath,session,skiplayer):
    '''
    加载模型，可以跳过某些层
    :param modelpath:模型文件的位置
    :param session:会话
    :param skiplayer:跳过的层
    :return:
    '''
    direct = np.load(modelpath,encoding='latin1').item()
    for key in direct:
        if key not in skiplayer:
            with tf.variable_scope(key,reuse=True):
               for subkey, data in direct[key].items():
                    session.run(tf.get_variable(subkey).assign(data))


def getPicslists(imgpaths):
    picList = []
    conts = os.listdir(imgpaths)
    for file in conts:
        filepath = os.path.join(imgpaths, file)
        if os.path.isfile(filepath):
            if filepath.__contains__("jpg") or filepath.__contains__("jpeg"):
                picList.append(filepath)
        else:
            picList.extend(getPicslists(imgpaths + os.sep + file))

    return picList

def onlyfiles(dir):
    conts = os.listdir(dir)
    for file in conts:
        filepath = os.path.join(dir,file)
        if os.path.isdir(filepath):
            return  False
    return True

def getPicsDrilists(imgpaths):
    picList = []
    conts = os.listdir(imgpaths)
    for file in conts:
        filepath = os.path.join(imgpaths, file)
        if os.path.isdir(filepath):
            if onlyfiles(filepath):
                picList.append(filepath)
            else:
                picList.extend(getPicsDrilists(filepath))
    return picList

def get_all_label(filepath):
    imglist = getPicsDrilists(filepath)
    labels = []
    for image in imglist:
        pathsp = image.split(os.sep)
        label = pathsp[-3]+'_'+pathsp[-2]+'_'+pathsp[-1]
        labels.append(label)
    return labels

def remove_invalidity_image(filepath):
    imglist = getPicslists(filepath)
    for image in imglist:
        try:
            Image.open(image).verify()
        except:
            os.remove(image)
            print(image)

if __name__ == '__main__':
    remove_invalidity_image(r"C:\Users\caopan.58GANJI-CORP\data\pic_after_filer")