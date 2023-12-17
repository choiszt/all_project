import tflearn
import tensorflow as tf
from functools import partial

def NN(inputs, inputs2):
    loss = inputs[:, :4]
    delay_interval = inputs[:, 4:]
    throughput = x_ = tflearn.reshape(inputs2, [-1, 10])
    loss=tf.layers.dense(loss,256,partial(tf.nn.leaky_relu, alpha=0.01))
    delay_interval = tf.layers.dense(delay_interval, 256, activation=partial(tf.nn.leaky_relu, alpha=0.01))
    throughput=tf.layers.dense(throughput,256,partial(tf.nn.leaky_relu, alpha=0.01))
    loss=tf.layers.dense(loss,512,partial(tf.nn.leaky_relu, alpha=0.01))
    delayinterval=tf.layers.dense(delay_interval,512,partial(tf.nn.leaky_relu, alpha=0.01))
    throughput=tf.layers.dense(throughput,512,partial(tf.nn.leaky_relu, alpha=0.01))
    lay3=tf.concat([loss,delayinterval,throughput],1)
    dplay3=tf.layers.dropout(lay3,0.5)
    lay4=tf.layers.dense(dplay3,64,partial(tf.nn.leaky_relu, alpha=0.01))
    output=tf.layers.dense(lay4,21,partial(tf.nn.leaky_relu, alpha=0.01))
    output=tf.layers.flatten(output)
    return output