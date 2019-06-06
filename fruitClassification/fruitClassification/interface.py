import numpy as np
import tensorflow as tf
import os
from PIL import Image
model_graph = tf.Graph()
with model_graph.as_default():
    od_graph_def = tf.GraphDef()
    # 读入Pb, 读图结构
    with tf.gfile.GFile('H:\\yan\\fruitClassification2\\datasets\\pbpath\\frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
with model_graph.as_default():
        with tf.Session(graph=model_graph) as sess:
            inputs = model_graph.get_tensor_by_name('image_tensor:0') 
            classes = model_graph.get_tensor_by_name('classes:0')
            #path = 'H:\\yan\\fruitClassification2\\datasets\\images'
            path = 'H:\\yan\\fruitClassification2\\datasets\\fruitTest64_48'
            index = -1
            true = 0
            # 读入路径下的所有图片
            for f in os.listdir(path):
                try:
                    label = int((path + '\\' + f).split('_')[-1].split('.')[0])
                    # 读入图片， 变为矩阵， 将图片输入到inputs容器， 得到Classes（分类结果）的值
                    im = Image.open(path + '\\' + f)
                    image = np.array(im, dtype=np.uint8)
                    image = image.reshape((image.shape[1], image.shape[0], image.shape[2]))
                    image_np = np.expand_dims(image, axis=0)
                    predicted_label = sess.run(classes, 
                                                feed_dict={inputs: image_np})
                    index += 1
                    if (predicted_label[0] == label):
                        true += 1
                        print("true{0}：/total{1}".format(true, index))
                    print(predicted_label[0], ' vs ', label)
                except Exception as e:
                    #im.show()
                    continue
            print(true / index)