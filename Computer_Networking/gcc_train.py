from network_simulator import receiver, sender, router, packet
from network_simulator.frame import Frame
from SendSideCongestionControl import SendSideCongestionController
import _thread
import os
import numpy as np
import warnings
import network
import tensorflow as tf
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import argparse#命令行
parser=argparse.ArgumentParser()
parser.add_argument('--load_model',action="store_true")
parser.add_argument('--UPDATE_SIZE',type=int,default=16)
parser.add_argument('--poly_lr',type=float,default=0.001)
parser.add_argument('--poly_endlr',type=float,default=0.0001)
parser.add_argument('--use_cycle',action="store_true")
parser.add_argument('--exp_lr',type=float,default=0.002)
# parser.add_argument('--UPDATE_SIZE',type=int,default=16)
args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
loss_window = 10
router_buffer = 20
start_bitrate_bps = 300 * 1000
stop_time = 60 * 1000 * 200
read_interval = 1000
default_frame_size = 30 * 1000
S_LEN = 4  # take how many frames in the past
start_arrival_time = 0

# 一次训练中的 batch 宽度
args.UPDATE_SIZE = 16
# 码率区间
ACTOR_VECTOR = np.arange(0.0, 2.02, 0.1)
ACTOR_SLOTS = len(ACTOR_VECTOR)


def get_predict_index(logits):  # 根据神经网络的输出得到最终预测结果
    ret = tf.cast(tf.argmax(logits, 1), tf.int32)
    return ret[0]


# 神经网络的训练，迭代神经网络的模型并更新参数
def get_evaluate_indicators(global_steps, logits, y_, batch_size,flag=0):
    distance_gradient = tf.zeros(batch_size)
    for i in range(0, batch_size):
        pre_index = tf.cast(tf.argmax(logits[i]), tf.int32)
        dis = pre_index - y_[i]
        one_hot = tf.one_hot(i, batch_size, dtype=np.float32)
        distance_gradient = distance_gradient + tf.cond(tf.greater(pre_index, y_[i]),
                                                        lambda: tf.cast(
                                                            tf.add(8, tf.pow(dis, 2)), tf.float32),
                                                        lambda: tf.cast(tf.pow(dis, 2), tf.float32)) * one_hot
    distance = tf.cast(tf.stop_gradient(distance_gradient), tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_, logits=logits)
    weight_loss = tf.reduce_mean(distance * loss)
    if(flag==1):
        lr = tf.train.polynomial_decay(
             learning_rate=args.poly_lr, global_step=global_steps, decay_steps=50,
            end_learning_rate=args.poly_endlr, power=0.5, cycle=args.use_cycle
)
    elif(flag==0):
        lr=tf.train.exponential_decay(args.exp_lr, global_steps, 3, 0.6, staircase=True)

    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(weight_loss)

    acc_one = 1 - tf.abs(tf.cast(tf.argmax(logits, 1),
                         tf.float32) - tf.cast(y_, tf.float32)) / ACTOR_SLOTS
    acc = tf.reduce_mean(acc_one)
    return weight_loss, train_op, acc, distance
losslist=[]
acclist=[]

def runable(x_train1, x_train2, y_train, train_op, loss, acc, logits, x1, x2, y_, distance, global_steps, batch_size, sess, epoch,losslist=losslist,acclist=acclist,):  # 将数据喂入神经网络
    # 训练和测试数据，可将n_epoch设置更大一些
    train_loss, train_acc,n_batch = 0, 0, 0
    _, err, ac, dist = sess.run([train_op, loss, acc, distance],
                                feed_dict={x1: x_train1, x2: x_train2, y_: y_train, global_steps: epoch})

    train_loss += err
    train_acc += ac
    n_batch += 1
    losslist.append(train_loss)
    acclist.append(train_acc)
    print("train loss: %f" % (train_loss / n_batch))
    print("train acc: %f" % (train_acc / n_batch))

if __name__ == '__main__':
    print("start")
    # ------------    仿真器初始化    ------------
    packet.Packet.set_max_packet_size(12000)  # bit
    Frame.set_fix_frame_size(False)
    Frame.set_default_frame_size(default_frame_size)  # bit
    sender = sender.Sender(start_bitrate_bps)
    receiver = receiver.Receiver(buffer_size=loss_window)
    router = router.Router(1800000, sender=sender,
                           receiver=receiver, buffer_size=router_buffer)
    sender.set_receiver(router)
    router.base_dir = './trace_data2/temp/mats1/'
    router.mat = './trace_data2/after760h.mat'
    router.read_interval = read_interval         # 限制带宽
    router.stop_time = stop_time    # 设置仿真器传输时间
    router.set_fix_bitrate(False)
    congestion_controller = SendSideCongestionController()
    congestion_controller.SetStartBitrate(start_bitrate_bps)
    target_send_rate = start_bitrate_bps
    rate, lbrates, dbrates, delay, loss1, delay_diff, bandwidth = [], [], [], [], [], [], []
    ts_delta, t_delta, trendline, mt, threashold = [], [], [], [], []
    send_time_last_state = 0
    frame_time_windows = []
    feedbackPacket = router.start(target_send_rate)
    # -----------------------------------------

    simulate_round = 0  # 仿真器模拟的轮数
    epoch = 0  # 神经网络训练的轮数
    predict_index = 5  # 初始预测码率在ACTOR_VECTOR的下标

    # loss & delay interval, 更新神经网络第一项输入
    obs_loss_and_interval_batch = np.zeros([args.UPDATE_SIZE, 8])
    # throughput, 更新神经网络第二项输入
    obs_throughput_batch = np.zeros([args.UPDATE_SIZE, 10])
    # 更新神经网络的标签
    gcc_label_batch = np.zeros([args.UPDATE_SIZE])

    # 仿真器一轮模拟的 loss 和 delay interval 结果
    # 一个 obs_loss_and_interval_batch 包含 UPDATE_SIZE 行 input_loss_and_interval
    input_loss_and_interval = np.zeros([1, 8])
    # 仿真器一轮模拟的 throughput 结果
    # 一个 obs_throughput_batch 包含 UPDATE_SIZE 行 input_throughput
    input_throughput = np.zeros([1, 10])
    no_optim = 0
    train_epoch_best_loss = 1000000
    # 模型saver初始化
    saver = tf.train.Saver()
    flag=0#标注是否需要缩小学习率
    with tf.Session() as sess:
        if(args.load_model==True):#模型读取
            new_saver = tf.train.import_meta_graph("./mymodel/bestmodel.meta")
            new_saver.restore(sess, "./mymodel/bestmodel")
            print('finish loading model!')
            graph = tf.get_default_graph()
            x_ = graph.get_tensor_by_name("x_ckpt:0")
            y_ = graph.get_tensor_by_name("y_:0")

        # input_loss_and_interval 以及 obs_batch 置入 x1
        x_loss_and_interval = tf.placeholder(
            tf.float32, shape=[None, 8], name='x_ckpt')
        # input_throughput以及 obs_throughput_batch 置入 x2
        x_throughput = tf.placeholder(
            tf.float32, shape=[None, 10], name='x_ckpt')
        # 仿真器内部gcc产生的标签置入y_gcc_label
        y_gcc_label = tf.placeholder(
            tf.int32, shape=[None, ], name='y_')  # 标签喂入y_
        # 设置神经网络的输出
        logits = network.NN(x_loss_and_interval, x_throughput)
        # 代表全局步数，比如在多少步该进行什么操作，现在神经网络训练到多少轮等等，类似于一个钟表。
        global_steps = tf.placeholder(tf.int32, shape=[], name='global_steps')

        # 计算神经网络评估指标
        # loss 损失函数
        # train_op 神经网络参数
        # acc 准确度
        # distance 距离
        loss, train_op, acc, distance = get_evaluate_indicators(global_steps=global_steps, logits=logits, y_=y_gcc_label,
                                                                batch_size=len(gcc_label_batch),flag=flag)
        # 神经网络最终预测值的下标
        index = get_predict_index(logits)  # 神经网络最终的输出预测值

        sess.run((tf.global_variables_initializer()))  # 初始化

        # 模型重新加载

        while(True):
            # ------------    仿真器指标获取    ------------
            bandwidth.append(feedbackPacket.average_bandwidth)
            target_bitrate, _, _, _, _, _, _, _ = congestion_controller.OnRTCPFeedbackPacket(
                feedbackPacket)
            loss1 = feedbackPacket.loss
            send_time = feedbackPacket.send_time_ms
            arrival_time = feedbackPacket.arrival_time_ms
            payload_size = feedbackPacket.payload_size
            average_bandwidth = float(
                feedbackPacket.average_bandwidth) / 1000000.0
            delay_send = []
            delay_arrival = []
            delay = []
            delay_interval = []
            arrival_time_last_state = arrival_time[-1]
            delay_interval.append(
                (send_time[0] - send_time_last_state) - (arrival_time[0] - arrival_time_last_state))
            for i in range(1, S_LEN):
                delay_send.append(send_time[i] - send_time[i-1])
                delay_arrival.append(arrival_time[i] - arrival_time[i-1])
                delay_interval.append(
                    delay_arrival[i-1] - delay_send[i-1])  # delay_interval
            send_time_last_state = send_time[-1]
            arrival_time_last_state = arrival_time[-1]
            for i in range(S_LEN):
                delay.append(arrival_time[i] - send_time[i])
            intervals = arrival_time[-1] - start_arrival_time
            throughput = np.sum(payload_size) / intervals / 1000
            start_arrival_time = arrival_time[-1]
            # -----------------------------------------

            # target_bitrate是仿真器内部gcc计算出来的目标码率，将其映射在ACTOR_VECTOR上
            # target_bitrate_index是映射结果的下标
            target_bitrate_index = int(
                target_bitrate / 100000 + 0.5)  # 将目标码率映射到相应的下表
            if (target_bitrate_index >= ACTOR_SLOTS):
                target_bitrate_index = ACTOR_SLOTS - 1  # 下标过大则映射为最大值

            # loss 和 delay_interval 填入input
            input_loss_and_interval[0][:4] = loss1[:4]
            input_loss_and_interval[0][4:] = delay_interval[:4]

            # 构造input_throughput
            if (simulate_round == 0):
                # 第一轮时将10行都设置为同一个throughput
                for i in range(0, 10):
                    input_throughput[0][i] = throughput
            else:
                # 每行上移，将最后一行设置为新的throughput模拟结果
                for i in range(0, 9):
                    input_throughput[0][i] = input_throughput[0][i + 1]
                input_throughput[0][9] = throughput

            # 在 batch 的相应位置填入 input 数据, 按照 unpdate_size 取模填充 batch
            obs_loss_and_interval_batch[simulate_round %
                                        args.UPDATE_SIZE] = input_loss_and_interval[0]
            gcc_label_batch[simulate_round %
                            args.UPDATE_SIZE] = target_bitrate_index
            obs_throughput_batch[simulate_round %
                                 args.UPDATE_SIZE] = input_throughput[0]

            # 收集够一个batch的数据进行神经网络的更新
            if (simulate_round + 1) % args.UPDATE_SIZE == 0:
                runable(x_train1=obs_loss_and_interval_batch, x_train2=obs_throughput_batch, y_train=gcc_label_batch, train_op=train_op,
                        loss=loss,
                        acc=acc, logits=logits, x1=x_loss_and_interval, x2=x_throughput, y_=y_gcc_label, distance=distance, global_steps=global_steps,
                        batch_size=32, sess=sess, epoch=epoch)
                epoch += 1
                print("epoch=",epoch)
                if(losslist[len(losslist)-1]>=train_epoch_best_loss):
                    no_optim+=1
                else:
                    no_optim=0
                    train_epoch_best_loss=losslist[len(losslist)-1]
                if no_optim>20:
                    print('early stop at %d epoch'%epoch)
                    saver.save(sess, "./mymodel/" + "bestmodel")
                    break
                if no_optim>10:
                    flag=1
                print(no_optim)


            # 得到神经网络的预测下标
            predict_index = sess.run(
                index, feed_dict={x_loss_and_interval: input_loss_and_interval, x_throughput: input_throughput})  # 得到神经网络的预测下标
            if predict_index == 0:
                predict_index += 1
            predict_bitrate = int(
                round(ACTOR_VECTOR[int(predict_index)] * 1e6))# 得到最终预测码率

            # 用预测码率进行仿真器的视频传输，同时得到feedbackPacket，进行下一轮的更新。
            feedbackPacket = router.start(predict_bitrate)
            simulate_round += 1
            # if simulate_round > 800:
            #     break
        import numpy as np
        import matplotlib.pyplot as plt
        import pylab as pl
        iter=[]
        for i in range(len(losslist)):
            iter.append(i*args.UPDATE_SIZE)
        fig = plt.figure(figsize=(7, 5))  # figsize是图片的大小`
        ax1 = fig.add_subplot(1, 1, 1)  # ax1是子图的名字`

        pl.plot(iter, losslist, 'g-', label=u'Loss')
        pl.legend()
        pl.xlabel(u'iters')
        pl.ylabel(u'loss')
        plt.show()
        pl.plot(iter, acclist, 'r-', label=u'acc')
        pl.legend()
        pl.xlabel(u'iters')
        pl.ylabel(u'acc')
        plt.show()


