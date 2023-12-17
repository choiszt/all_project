import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
def one_hot(y):
    res = np.zeros([y.shape[0], 10])

    for i in range(y.shape[0]):
        res[i][y[i]] = 1
    return res

def load_data():
    train_set_x_orig = np.load('../data/train_data.npy', encoding='bytes')  # train set features (60000,784)
    train_set_y_orig = np.load('../data/train_label.npy', encoding='bytes')  # train set labels (60000,1)
    test_set_x_orig = np.load('../data/test_data.npy', encoding='bytes')  # test set features (10000,784)
    test_set_y_orig = np.load('../data/test_label.npy', encoding='bytes')  # test set labels (10000,1)
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

class Linear(object):#定义线性运算
    def __init__(self, n_in, n_out):
        self.input = None
        # kaiming normal
        self.W = np.random.normal(
            loc=0,  #均值
            scale=2 / n_in,  #标准差
            size=(n_in, n_out)) #输出的shape

        self.b = np.zeros(n_out, )

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        #optimizer的参数
        self.m_W = self.dW
        self.m_b = self.db
        self.v_W =self.dW
        self.v_b = self.db

    def forward(self, input):
        self.input = input
        lin_output = np.dot(input, self.W) + self.b  #x*W+b
        self.Z = lin_output
        return self.Z

    __call__ = forward

    def backward(self, dZ, output_layer=False):#dz是y和y_hat的差
        self.dW = np.atleast_2d(self.input).T.dot(np.atleast_2d(dZ)) / dZ.shape[0]
        self.db = np.mean(dZ)
        self.dA = dZ.dot(self.W.T)
        return self.dA

class ReLU():
    def __init__(self):
        self.A = None
        self.Z = None

    def forward(self, Z):
        self.A = np.maximum(0,Z)
        assert(self.A.shape == Z.shape)
        self.Z = Z
        return self.A
    __call__ = forward

    def backward(self, dA):
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        # When z <= 0, set dz to 0 as well.
        dZ[self.Z <= 0] = 0
        return dZ

class Softmax():
    def forward(self, Z):
        t = np.sum(np.exp(Z), axis=1)
        AL = np.exp(Z) / t[:,np.newaxis]
        return AL
    __call__ = forward


class Dropout():
    def __init__(self, dropout_rate=0.5, is_training=True):
        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError("dropout probability has to be between 0 and 1,"
                             "but got {}".format(dropout_rate))
        dropout_rate = 1 - dropout_rate
        self.dropout_rate = dropout_rate
        self.is_training = is_training

    def forward(self, A):
        self.A = np.array(A, copy=True)
        if self.is_training:
            self.binary_scaled_mask = np.random.binomial(1, self.dropout_rate,
                                                         size=self.A.shape) / self.dropout_rate
            #相当于一次trail中，留下概率为droout_rate
            self.A *= self.binary_scaled_mask
        return self.A

    __call__ = forward

    def backward(self, dA):
        dA *= self.binary_scaled_mask
        return dA


class BN():

    def __init__(self, n_out, momentum_BN=0.9):

        self.gamma = np.ones(n_out)
        self.beta = np.zeros(n_out)

        self.dgamma = np.zeros(self.gamma.shape)
        self.dbeta = np.zeros(self.beta.shape)
        self.m_gamma = np.zeros(self.gamma.shape)
        self.v_gamma = np.zeros(self.gamma.shape)
        self.m_beta = np.zeros(self.beta.shape)
        self.v_beta = np.zeros(self.beta.shape)
        self.momentum_BN = momentum_BN
        self.mean = 0.
        self.var = 0.
        self.mean_avg = self.mean
        self.var_avg = self.var
        self.epsilon = 1e-7

        self.is_training = True

    def forward(self, Z):
        self.Z = Z
        mean = Z.mean(axis=0)
        self.mean = mean
        self.mean_avg = (1. - self.momentum_BN) * self.mean_avg + self.momentum_BN * mean #加入momentum参数
        # EMA
        var = Z.var(axis=0)
        self.var = var
        self.var_avg = (1. - self.momentum_BN) * self.var_avg + self.momentum_BN * var

        if self.is_training:
            self.Z_hat = (self.Z - mean) / np.sqrt(var + self.epsilon)
        else:
            self.Z_hat = (self.Z - self.mean_avg) / np.sqrt(self.var_avg + self.epsilon)
        output = self.gamma * self.Z_hat + self.beta
        return output

    __call__ = forward

    def backward(self, dA):
        self.dgamma = np.sum(dA * self.Z_hat, axis=0)
        self.dbeta = np.sum(dA, axis=0)
        dZ_hat = dA * self.gamma
        dsigma = -0.5 * np.power(self.var + self.epsilon, -1.5) * np.sum(dZ_hat * (self.Z - self.mean), axis=0)
        dmu = -np.sum(dZ_hat / np.sqrt(self.var + self.epsilon), axis=0) - 2 * dsigma * np.sum(self.Z - self.mean,
                                                                                               axis=0) / self.Z.shape[0]
        dA = dZ_hat / np.sqrt(self.var + self.epsilon) + 2. * dsigma * (self.Z - self.mean) / self.Z.shape[0] + dmu / \
             self.Z.shape[0]
        return dA


class nn_MLP:
    def __init__(self, layers, activations, config):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(Linear(layers[i], layers[i + 1]))
            if config[1]:#如果为none则layer不添加
                self.layers.append(BN(layers[i + 1], config[1]))
            if activations[i + 1].lower() == 'relu':
                act_layer = ReLU()
            elif activations[i + 1].lower() == 'softmax':
                act_layer = Softmax()
            self.layers.append(act_layer)
            if config[0]:
                self.layers.append(Dropout(config[0]))
        if config[0]:
            self.layers.pop(-1)
        # print(self.layers)

    def forward(self, input, mode):
        output = None
        for layer in self.layers:
            if not isinstance(layer, Linear):
                if mode == "test":
                    layer.is_training = False
                else:
                    layer.is_training = True
            output = layer(input)
            input = output
        AL = output
        return AL

    __call__ = forward

class CrossEntropyLoss:
    def __init__(self, model):
        self.model = model
        return

    def loss(self, y, AL):

        self.y = one_hot(y)
        self.AL = AL
        id0 = range(y.shape[0]) # range of examples
        # Compute loss from AL and y.
        cost = -np.mean(np.sum((self.y * np.log(self.AL + 1e-6)), axis=1, keepdims=True))
        cost = np.squeeze(cost)
        return cost

    def backward(self):
        dZ = self.AL - self.y
        d = dZ
        for layer in reversed(self.model.layers[:-1]):

            d = layer.backward(d)#倒着逆向求每层的backward


class SGDOptimizer:
    def __init__(self, model, lr=0.001, momentum=0.9, weight_decay=0.01):
        self.lr = lr
        self.model = model
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self):
        for layer in self.model.layers:
            if isinstance(layer, Linear):
                v_W = self.momentum * layer.v_W + self.lr * layer.dW
                v_b = self.momentum * layer.v_b + self.lr * layer.db

                layer.W = layer.W - v_W
                layer.W = layer.W - self.lr * layer.W * self.weight_decay
                layer.b = layer.b - v_b

                layer.v_W = v_W
                layer.v_b = v_b

            elif isinstance(layer, BN):
                v_gamma = self.momentum * layer.v_gamma + self.lr * layer.dgamma
                v_beta = self.momentum * layer.v_beta + self.lr * layer.dbeta

                layer.gamma-=v_gamma
                layer.gamma-=self.lr * layer.gamma * self.weight_decay
                layer.beta-= v_beta

                layer.v_gamma = v_gamma
                layer.v_beta = v_beta


class Adam:
    def __init__(self, model, lr=0.001, betas=(0.9, 0.999), epsilon=1e-8):
        self.lr = lr
        self.model = model
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.epsilon = epsilon
        self.iter = 0

    def step(self):
        for layer in self.model.layers:
            if isinstance(layer, Linear):
                self.iter += 1
                layer.m_W = (self.beta1 * layer.m_W + (1 - self.beta1) * layer.dW)
                layer.v_W = (self.beta2 * layer.v_W + (1 - self.beta2) * (layer.dW ** 2))
                m_hat_W = layer.m_W / (1 - self.beta1 ** self.iter)
                v_hat_W = layer.v_W / (1 - self.beta2 ** self.iter)
                layer.W -= (self.lr / (np.sqrt(v_hat_W + self.epsilon)) * m_hat_W)

                layer.m_b = (self.beta1 * layer.m_b + (1 - self.beta1) * layer.db)
                layer.v_b = (self.beta2 * layer.v_b + (1 - self.beta2) * (layer.db ** 2))
                m_hat_b = layer.m_b / (1 - self.beta1 ** self.iter)
                v_hat_b = layer.v_b / (1 - self.beta2 ** self.iter)
                layer.b -= (self.lr / (np.sqrt(v_hat_b + self.epsilon)) * m_hat_b)
            elif isinstance(layer, BN):
                layer.m_gamma = (self.beta1 * layer.m_gamma + (1 - self.beta1) * layer.dgamma)
                layer.v_gamma = (self.beta2 * layer.v_gamma + (1 - self.beta2) * (layer.dgamma ** 2))
                m_hat_gamma = layer.m_gamma / (1 - self.beta1 ** self.iter)
                v_hat_gamma = layer.v_gamma / (1 - self.beta2 ** self.iter)
                layer.gamma -= (self.lr / (np.sqrt(v_hat_gamma + self.epsilon)) * m_hat_gamma)

                layer.m_beta = (self.beta1 * layer.m_beta + (1 - self.beta1) * layer.dbeta)
                layer.v_beta = (self.beta2 * layer.v_beta + (1 - self.beta2) * (layer.dbeta ** 2))
                m_hat_beta = layer.m_beta / (1 - self.beta1 ** self.iter)
                v_hat_beta = layer.v_beta / (1 - self.beta2 ** self.iter)
                layer.beta -= (self.lr / (np.sqrt(v_hat_beta + self.epsilon)) * m_hat_beta)

train_x, train_y, test_x, test_y = load_data()
# train_x, test_x = preprocessing(train_x, test_x)


def train_epoch(model, loss_fn, batch_size, epoch, opt, model_name):
    N = train_x.shape[0]
    mix_ids = np.random.permutation(N)  # mix data
    nbatches = int(np.ceil(float(N) / batch_size))
    loss = np.zeros(nbatches)
    for beg_i in tqdm(range(nbatches), desc="Epoch {} for model {}".format(epoch + 1, model_name)):
        # get the i-th batch
        batch_ids = mix_ids[batch_size * beg_i:min(batch_size * (beg_i + 1), N)]
        x_batch, y_batch = train_x[batch_ids], train_y[batch_ids]

        # forward pass
        y_hat = model(x_batch, "train")
        # backward pass
        loss[beg_i] = loss_fn.loss(y_batch, y_hat)
        loss_fn.backward()

        # update
        opt.step()
        # if beg_i % 100 == 0:
        #     print("Loss:{:7.4f} [{}/{}]".format(loss[beg_i], beg_i * len(x_batch), train_x.shape[0]))
        #     # time.sleep(0.1)
    return loss

def test_train(model, loss_fn, batch_size, epoch):
    correct = 0
    loss = 0
    nbatches = int(np.ceil(float(train_x.shape[0]) / batch_size))
    losses = np.zeros(nbatches)
    ids = np.arange(train_x.shape[0])
    for it in range(nbatches):
        batch_ids = ids[batch_size * it:min(batch_size * (it + 1), train_x.shape[0])]
        X_batch, y_batch = train_x[batch_ids], train_y[batch_ids]

        outputs = model(X_batch, "test")
        loss += loss_fn.loss(y_batch, outputs)
        losses[it] = loss_fn.loss(y_batch, outputs)
        correct += (outputs.argmax(1) == one_hot(y_batch).argmax(1)).astype(int).sum()
    loss /= nbatches

    print("Evaluation on training set:\n      Accuracy:{:4.2f}%, Avg loss:{:10.7f}".format(correct / train_x.shape[0] * 100, loss))

    return correct / train_x.shape[0] * 100, losses

def test_val(model, loss_fn, batch_size, epoch):
    correct = 0
    loss = 0
    nbatches = int(np.ceil(float(test_x.shape[0]) / batch_size))
    losses = np.zeros(nbatches)
    ids = np.arange(test_x.shape[0])
    for it in range(nbatches):
        batch_ids = ids[batch_size * it:min(batch_size * (it + 1), test_x.shape[0])]
        X_batch, y_batch = test_x[batch_ids], test_y[batch_ids]
        outputs = model(X_batch, "test")
        loss += loss_fn.loss(y_batch, outputs)
        losses[it] = loss_fn.loss(y_batch, outputs)
        correct += (outputs.argmax(1) == one_hot(y_batch).argmax(1)).astype(int).sum()
    loss /= nbatches

    print("Evaluation on testing set:\n      Accuracy:{:4.2f}%, Avg loss:{:10.7f}".format(correct / test_x.shape[0] * 100, loss))

    return correct / test_x.shape[0] * 100, losses

#改变层数
# # params
# learning_rate = 1e-3
# epochs = 25
# batch_size = 32
#
# # 1 layer
# model1 = nn_MLP([784, 128, 10], ['None', 'ReLU', 'Softmax'], (0.2, 0.9))
# opt1 = Adam(model1, lr=learning_rate)
# loss_fn1 = CrossEntropyLoss(model1)
# losses1 = np.zeros(epochs)
# hidden_acc1 = []
# hidden_loss1 = []
#
# # 2 layers
# model2 = nn_MLP([784, 128, 128, 10], ['None','ReLU', 'ReLU', 'Softmax'], (0.2, 0.9))
# opt2 = Adam(model2, lr=learning_rate)
# loss_fn2 = CrossEntropyLoss(model2)
# losses2 = np.zeros(epochs)
# hidden_acc2 = []
# hidden_loss2 = []
#
# # 3 layers
# model3 = nn_MLP([784, 128, 128, 128, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.2, 0.9))
# opt3 = Adam(model3, lr=learning_rate)
# loss_fn3 = CrossEntropyLoss(model3)
# losses3 = np.zeros(epochs)
# hidden_acc3 = []
# hidden_loss3 = []
#
#
# # 5 layers
# model4 = nn_MLP([784, 128, 128, 128, 128, 128, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.2, 0.9))
# opt4 = Adam(model4, lr=learning_rate)
# loss_fn4 = CrossEntropyLoss(model4)
# losses4 = np.zeros(epochs)
# hidden_acc4 = []
# hidden_loss4 = []
#
#
# print("Train start ---------------------- ")
# for k in range(epochs):
#     # main model
#     loss1 = train_epoch(model1, loss_fn1, batch_size, k, opt1, "1")
#     acc_train1, _ = test_train(model1, loss_fn1, batch_size, k)
#     acc_val1, val_loss1 = test_val(model1, loss_fn1, batch_size, k)
#     losses1[k] = np.mean(loss1)
#     hidden_acc1.append(acc_val1)
#     hidden_loss1.append(val_loss1.mean())
#
#
#     #side model2
#     loss2 = train_epoch(model2, loss_fn2, batch_size, k, opt2, "2")
#     acc_train2, _ = test_train(model2, loss_fn2, batch_size, k)
#     acc_val2, val_loss2 = test_val(model2, loss_fn2, batch_size, k)
#     losses2[k] = np.mean(loss2)
#     hidden_acc2.append(acc_val2)
#     hidden_loss2.append(val_loss2.mean())
#
#     # side model3
#     loss3 = train_epoch(model3, loss_fn3, batch_size, k, opt3, "3")
#     acc_train3, _ = test_train(model3, loss_fn3, batch_size, k)
#     acc_val3, val_loss3 = test_val(model3, loss_fn3, batch_size, k)
#     losses3[k] = np.mean(loss3)
#     hidden_acc3.append(acc_val3)
#     hidden_loss3.append(val_loss3.mean())
#
#     # side model3
#     loss4 = train_epoch(model4, loss_fn4, batch_size, k, opt4, "4")
#     acc_train4, _ = test_train(model4, loss_fn4, batch_size, k)
#     acc_val4, val_loss4 = test_val(model4, loss_fn4, batch_size, k)
#     losses4[k] = np.mean(loss4)
#     hidden_acc4.append(acc_val4)
#     hidden_loss4.append(val_loss4.mean())
#
#     # tensorboard record
#     writer.add_scalars("num_layers/val_loss", {"1 layer": val_loss1.mean(), "2 layers": val_loss2.mean(), "3 layers": val_loss3.mean(), "5 layers": val_loss4.mean()}, k)
#     writer.add_scalars("num_layers/val_accuracy", {"1 layer": acc_val1, "2 layers": acc_val2, "3 layers":acc_val3, "5 layers": acc_val4}, k)
#
#
# # flush the records
# writer.flush()
# writer.close()
#
# print("Done!")
#
# # %load_ext tensorboard
# # %tensorboard --logdir=runs
#
#
# # print losses
# plt.figure(figsize=(6,3))
# plt.title('Loss Plot')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.plot(hidden_loss1, label="1 layer")
# plt.plot(hidden_loss2, label="2 layers")
# plt.plot(hidden_loss3, label="3 layers")
# plt.plot(hidden_loss4, label="5 layers")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()
#
# # print accuracy
# plt.figure(figsize=(6,3))
# plt.title('Accuracy Plot')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.plot(hidden_acc1, label="1 layer")
# plt.plot(hidden_acc2, label="2 layers")
# plt.plot(hidden_acc3, label="3 layers")
# plt.plot(hidden_acc4, label="5 layers")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()

#改变隐藏层神经元个数
# params
# learning_rate = 1e-3
# epochs = 3
# batch_size = 32
# # size64
# model1 = nn_MLP([784, 64, 64, 64, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.2, 0.9))
# opt1 = Adam(model1, lr=learning_rate)
# loss_fn1 = CrossEntropyLoss(model1)
# losses1 = np.zeros(epochs)
# size_acc1 = []
# size_loss1 = []
#
# # size128
# model2 = nn_MLP([784, 128, 128, 128, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.2, 0.9))
# opt2 = Adam(model2, lr=learning_rate)
# loss_fn2 = CrossEntropyLoss(model2)
# losses2 = np.zeros(epochs)
# size_acc2 = []
# size_loss2 = []
#
# # size256
# model3 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.2, 0.9))
# opt3 = Adam(model3, lr=learning_rate)
# loss_fn3 = CrossEntropyLoss(model3)
# losses3 = np.zeros(epochs)
# size_acc3 = []
# size_loss3 = []
#
# print("Train start ---------------------- ")
# for k in range(epochs):
#
#     # main model
#     loss1 = train_epoch(model1, loss_fn1, batch_size, k, opt1, "1")
#     acc_train1, _ = test_train(model1, loss_fn1, batch_size, k)
#     acc_val1, val_loss1 = test_val(model1, loss_fn1, batch_size, k)
#     losses1[k] = np.mean(loss1)
#     size_acc1.append(acc_val1)
#     size_loss1.append(val_loss1.mean())
#
#
#     #side model2
#     loss2 = train_epoch(model2, loss_fn2, batch_size, k, opt2, "2")
#     acc_train2, _ = test_train(model2, loss_fn2, batch_size, k)
#     acc_val2, val_loss2 = test_val(model2, loss_fn2, batch_size, k)
#     losses2[k] = np.mean(loss2)
#     size_acc2.append(acc_val2)
#     size_loss2.append(val_loss2.mean())
#
#     # side model3
#     loss3 = train_epoch(model3, loss_fn3, batch_size, k, opt3, "3")
#     acc_train3, _ = test_train(model3, loss_fn3, batch_size, k)
#     acc_val3, val_loss3 = test_val(model3, loss_fn3, batch_size, k)
#     losses3[k] = np.mean(loss3)
#     size_acc3.append(acc_val3)
#     size_loss3.append(val_loss3.mean())
#
#
#     #tensorboard record
#     writer.add_scalars("hidden_size/val_loss", {"size 64": val_loss1.mean(), "size 128": val_loss2.mean(), "size 256": val_loss3.mean()}, k)
#     writer.add_scalars("hidden_size/val_accuracy", {"size 64": acc_val1, "size 128": acc_val2, "size 256":acc_val3}, k)
#
#
# #flush the records
# writer.flush()
# writer.close()
#
# print("Done!")
#
# # %load_ext tensorboard
# # %tensorboard --logdir=runs
#
# # print losses
# plt.figure(figsize=(6,3))
# plt.title('Loss Plot')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.plot(size_loss1, label="size 64")
# plt.plot(size_loss2, label="size 128")
# plt.plot(size_loss3, label="size 256")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()
#
# # print accuracy
# plt.figure(figsize=(6,3))
# plt.title('Accuracy Plot')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.plot(size_acc1, label="size 64")
# plt.plot(size_acc2, label="size 128")
# plt.plot(size_acc3, label="size 256")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()

#改变学习率
# # params
# epochs = 3
# batch_size = 32
# # 1e-4
# model1 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.2, 0.9))
# opt1 = Adam(model1, lr=1e-4)
# loss_fn1 = CrossEntropyLoss(model1)
# losses1 = np.zeros(epochs)
# lr_acc1 = []
# lr_loss1 = []
#
# # 5e-4
# model2 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.2, 0.9))
# opt2 = Adam(model2, lr=5e-4)
# loss_fn2 = CrossEntropyLoss(model2)
# losses2 = np.zeros(epochs)
# lr_acc2 = []
# lr_loss2 = []
#
# # 1e-3
# model3 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.2, 0.9))
# opt3 = Adam(model3, lr=1e-3)
# loss_fn3 = CrossEntropyLoss(model3)
# losses3 = np.zeros(epochs)
# lr_acc3 = []
# lr_loss3 = []
#
# print("Train start ---------------------- ")
# for k in range(epochs):
#
#     # main model
#     loss1 = train_epoch(model1, loss_fn1, batch_size, k, opt1, "1")
#     acc_train1, _ = test_train(model1, loss_fn1, batch_size, k)
#     acc_val1, val_loss1 = test_val(model1, loss_fn1, batch_size, k)
#     losses1[k] = np.mean(loss1)
#     lr_acc1.append(acc_val1)
#     lr_loss1.append(val_loss1.mean())
#
#     #side model2
#     loss2 = train_epoch(model2, loss_fn2, batch_size, k, opt2, "2")
#     acc_train2, _ = test_train(model2, loss_fn2, batch_size, k)
#     acc_val2, val_loss2 = test_val(model2, loss_fn2, batch_size, k)
#     losses2[k] = np.mean(loss2)
#     lr_acc2.append(acc_val2)
#     lr_loss2.append(val_loss2.mean())
#
#     # side model3
#     loss3 = train_epoch(model3, loss_fn3, batch_size, k, opt3, "3")
#     acc_train3, _ = test_train(model3, loss_fn3, batch_size, k)
#     acc_val3, val_loss3 = test_val(model3, loss_fn3, batch_size, k)
#     losses3[k] = np.mean(loss3)
#     lr_acc3.append(acc_val3)
#     lr_loss3.append(val_loss3.mean())
#
#
#     # tensorboard record
#     writer.add_scalars("learning_rate/val_loss", {"1e-4": val_loss1.mean(), "5e-4": val_loss2.mean(), "1e-3": val_loss3.mean()}, k)
#     writer.add_scalars("learning_rate/val_accuracy", {"1e-4": acc_val1, "5e-4": acc_val2, "1e-3":acc_val3}, k)
#
#
# # flush the records
# writer.flush()
# writer.close()
#
# print("Done!")
#
# # %load_ext tensorboard
# # %tensorboard --logdir=runs
#
# # print losses
# plt.figure(figsize=(6,3))
# plt.title('Loss Plot')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.plot(lr_loss1, label="1e-4")
# plt.plot(lr_loss2, label="5e-4")
# plt.plot(lr_loss3, label="1e-3")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()
#
# # print accuracy
# plt.figure(figsize=(6,3))
# plt.title('Accuracy Plot')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.plot(lr_acc1, label="1e-4")
# plt.plot(lr_acc2, label="5e-4")
# plt.plot(lr_acc3, label="1e-3")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()

#改变batchsize大小
# params
# learning_rate = 5e-4
# epochs = 3
# # 1e-4
# model1 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.2, 0.9))
# opt1 = Adam(model1, lr=learning_rate)
# loss_fn1 = CrossEntropyLoss(model1)
# losses1 = np.zeros(epochs)
# batch_size1 = 16
# batch_acc1 = []
# batch_loss1 = []
#
# # 5e-4
# model2 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.2, 0.9))
# opt2 = Adam(model2, lr=learning_rate)
# loss_fn2 = CrossEntropyLoss(model2)
# losses2 = np.zeros(epochs)
# batch_size2 = 32
# batch_acc2 = []
# batch_loss2 = []
#
# # 1e-3
# model3 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.2, 0.9))
# opt3 = Adam(model3, lr=learning_rate)
# loss_fn3 = CrossEntropyLoss(model3)
# losses3 = np.zeros(epochs)
# batch_size3 = 54
# batch_acc3 = []
# batch_loss3 = []
#
# print("Train start ---------------------- ")
# for k in range(epochs):
#     # main model
#     loss1 = train_epoch(model1, loss_fn1, batch_size1, k, opt1, "1")
#     acc_train1, _ = test_train(model1, loss_fn1, batch_size1, k)
#     acc_val1, val_loss1 = test_val(model1, loss_fn1, batch_size1, k)
#     losses1[k] = np.mean(loss1)
#     batch_acc1.append(acc_val1)
#     batch_loss1.append(val_loss1.mean())
#
#     # side model2
#     loss2 = train_epoch(model2, loss_fn2, batch_size2, k, opt2, "2")
#     acc_train2, _ = test_train(model2, loss_fn2, batch_size2, k)
#     acc_val2, val_loss2 = test_val(model2, loss_fn2, batch_size2, k)
#     losses2[k] = np.mean(loss2)
#     batch_acc2.append(acc_val2)
#     batch_loss2.append(val_loss2.mean())
#
#     # side model3
#     loss3 = train_epoch(model3, loss_fn3, batch_size3, k, opt3, "3")
#     acc_train3, _ = test_train(model3, loss_fn3, batch_size3, k)
#     acc_val3, val_loss3 = test_val(model3, loss_fn3, batch_size3, k)
#     losses3[k] = np.mean(loss3)
#     batch_acc3.append(acc_val3)
#     batch_loss3.append(val_loss3.mean())
#
#     # tensorboard record
#     writer.add_scalars("batch_size/val_loss", {"batch16": val_loss1.mean(), "batch32": val_loss2.mean(), "batch64": val_loss3.mean()}, k)
#     writer.add_scalars("batch_size/val_accuracy", {"batch16": acc_val1, "batch32": acc_val2, "batch64":acc_val3}, k)
#
# # flush the records
# writer.flush()
# writer.close()
#
# print("Done!")
#
# # %load_ext tensorboard
# # %tensorboard --logdir=runs
#
# # print losses
# plt.figure(figsize=(6, 3))
# plt.title('Loss Plot')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.plot(batch_loss1, label="16")
# plt.plot(batch_loss2, label="32")
# plt.plot(batch_loss3, label="64")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()
#
# # print accuracy
# plt.figure(figsize=(6, 3))
# plt.title('Accuracy Plot')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.plot(batch_acc1, label="16")
# plt.plot(batch_acc2, label="32")
# plt.plot(batch_acc3, label="64")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()

# #改变dropout参数
# # params
# learning_rate = 5e-4
# batch_size = 32
# epochs = 3
# # dropout0.1
# model1 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.1, 0.9))
# opt1 = Adam(model1, lr=learning_rate)
# loss_fn1 = CrossEntropyLoss(model1)
# losses1 = np.zeros(epochs)
# dp_acc1 = []
# dp_loss1 = []
#
# # dropout0.3
# model2 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.3, 0.9))
# opt2 = Adam(model2, lr=learning_rate)
# loss_fn2 = CrossEntropyLoss(model2)
# losses2 = np.zeros(epochs)
# dp_acc2 = []
# dp_loss2 = []
#
# # dropout0.5
# model3 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.5, 0.9))
# opt3 = Adam(model3, lr=learning_rate)
# loss_fn3 = CrossEntropyLoss(model3)
# losses3 = np.zeros(epochs)
# dp_acc3 = []
# dp_loss3 = []
#
# print("Train start ---------------------- ")
# for k in range(epochs):
#
#     # main model
#     loss1 = train_epoch(model1, loss_fn1, batch_size, k, opt1, "1")
#     acc_train1, _ = test_train(model1, loss_fn1, batch_size, k)
#     acc_val1, val_loss1 = test_val(model1, loss_fn1, batch_size, k)
#     losses1[k] = np.mean(loss1)
#     dp_acc1.append(acc_val1)
#     dp_loss1.append(val_loss1.mean())
#
#
#     #side model2
#     loss2 = train_epoch(model2, loss_fn2, batch_size, k, opt2, "2")
#     acc_train2, _ = test_train(model2, loss_fn2, batch_size, k)
#     acc_val2, val_loss2 = test_val(model2, loss_fn2, batch_size, k)
#     losses2[k] = np.mean(loss2)
#     dp_acc2.append(acc_val2)
#     dp_loss2.append(val_loss2.mean())
#
#     # side model3
#     loss3 = train_epoch(model3, loss_fn3, batch_size, k, opt3, "3")
#     acc_train3, _ = test_train(model3, loss_fn3, batch_size, k)
#     acc_val3, val_loss3 = test_val(model3, loss_fn3, batch_size, k)
#     losses3[k] = np.mean(loss3)
#     dp_acc3.append(acc_val3)
#     dp_loss3.append(val_loss3.mean())
#
#
#     # tensorboard record
#     writer.add_scalars("dropout/val_loss", {"dropout0.1": val_loss1.mean(), "dropout0.2": val_loss2.mean(), "dropout0.5": val_loss3.mean()}, k)
#     writer.add_scalars("dropout/val_accuracy", {"dropout0.1": acc_val1, "dropout0.2": acc_val2, "dropout0.5":acc_val3}, k)
#
#
# # flush the records
# writer.flush()
# writer.close()
#
# print("Done!")
#
# # %load_ext tensorboard
# # %tensorboard --logdir=runs
# # print losses
# plt.figure(figsize=(6,3))
# plt.title('Loss Plot')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.plot(dp_loss1, label="0.1")
# plt.plot(dp_loss2, label="0.3")
# plt.plot(dp_loss3, label="0.5")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()
#
# # print accuracy
# plt.figure(figsize=(6,3))
# plt.title('Accuracy Plot')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.plot(dp_acc1, label="0.1")
# plt.plot(dp_acc2, label="0.3")
# plt.plot(dp_acc3, label="0.5")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()

# #optimizer比较
# # params
# learning_rate = 5e-4
# batch_size = 64
# epochs = 3
#
# # Adam
# model1 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.3, 0.9))
# opt1 = Adam(model1, lr=learning_rate)
# loss_fn1 = CrossEntropyLoss(model1)
# losses1 = np.zeros(epochs)
# opt_acc1 = []
# opt_loss1 = []
#
# # SGD with momentum
# model2 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.3, 0.9))
# opt2 = SGDOptimizer(model2, lr=learning_rate)
# loss_fn2 = CrossEntropyLoss(model2)
# losses2 = np.zeros(epochs)
# opt_acc2 = []
# opt_loss2 = []
#
# print("Train start ---------------------- ")
# for k in range(epochs):
#
#     # main model
#     loss1 = train_epoch(model1, loss_fn1, batch_size, k, opt1, "1")
#     acc_train1, _ = test_train(model1, loss_fn1, batch_size, k)
#     acc_val1, val_loss1 = test_val(model1, loss_fn1, batch_size, k)
#     losses1[k] = np.mean(loss1)
#     opt_acc1.append(acc_val1)
#     opt_loss1.append(val_loss1.mean())
#
#
#     #side model2
#     loss2 = train_epoch(model2, loss_fn2, batch_size, k, opt2, "2")
#     acc_train2, _ = test_train(model2, loss_fn2, batch_size, k)
#     acc_val2, val_loss2 = test_val(model2, loss_fn2, batch_size, k)
#     losses2[k] = np.mean(loss2)
#     opt_acc2.append(acc_val2)
#     opt_loss2.append(val_loss2.mean())
#
#     # tensorboard record
#     writer.add_scalars("optimiser/val_loss", {"Adam": val_loss1.mean(), "SGD": val_loss2.mean()}, k)
#     writer.add_scalars("optimiser/val_accuracy", {"Adam": acc_val1, "SGD": acc_val2}, k)
#
#
# # flush the records
# writer.flush()
# writer.close()
#
# print("Done!")
#
# # %load_ext tensorboard
# # %tensorboard --logdir=runs
#
# plt.figure(figsize=(6,3))
# plt.title('Loss Plot')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.plot(opt_loss1, label="Adam")
# plt.plot(opt_loss2, label="SGD")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()
#
# # print accuracy
# plt.figure(figsize=(6,3))
# plt.title('Accuracy Plot')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.plot(opt_acc1, label="Adam")
# plt.plot(opt_acc2, label="SGD")
# plt.legend(loc="upper left")
# plt.grid()
# plt.show()

#消融实验
# params
learning_rate = 5e-4
batch_size = 64
epochs = 3

#无dropout
model1 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (None, 0.9))
opt1 = Adam(model1, lr=learning_rate)
loss_fn1 = CrossEntropyLoss(model1)
losses1 = np.zeros(epochs)
abl_acc1 = []
abl_loss1 = []
#无BN层
model2 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.3, None))
opt2 = Adam(model2, lr=learning_rate)
loss_fn2 = CrossEntropyLoss(model2)
losses2 = np.zeros(epochs)
abl_acc2 = []
abl_loss2 = []

#无dropout无BN
model3 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (None, None))
opt3 = Adam(model3, lr=learning_rate)
loss_fn3 = CrossEntropyLoss(model3)
losses3 = np.zeros(epochs)
abl_acc3 = []
abl_loss3 = []

#baseline
model4 = nn_MLP([784, 256, 256, 256, 10], ['None', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], (0.3, 0.9))
opt4 = Adam(model4, lr=learning_rate)
loss_fn4 = CrossEntropyLoss(model4)
losses4 = np.zeros(epochs)
abl_acc4 = []
abl_loss4 = []

# Ablation studies
print("Train start ---------------------- ")



for k in range(epochs):

    # No dp
    loss1 = train_epoch(model1, loss_fn1, batch_size, k, opt1, "1")
    acc_train1, train_loss1 = test_train(model1, loss_fn1, batch_size, k)
    acc_val1, val_loss1 = test_val(model1, loss_fn1, batch_size, k)
    losses1[k] = np.mean(loss1)
    abl_acc1.append(acc_val1)
    abl_loss1.append(val_loss1.mean())

    # No BN
    loss2 = train_epoch(model2, loss_fn2, batch_size, k, opt2, "2")
    acc_train2, train_loss2 = test_train(model2, loss_fn2, batch_size, k)
    acc_val2, val_loss2 = test_val(model2, loss_fn2, batch_size, k)
    losses2[k] = np.mean(loss2)
    abl_acc2.append(acc_val2)
    abl_loss2.append(val_loss2.mean())

    # None
    loss3 = train_epoch(model3, loss_fn3, batch_size, k, opt3, "3")
    acc_train3, train_loss3 = test_train(model3, loss_fn3, batch_size, k)
    acc_val3, val_loss3 = test_val(model3, loss_fn3, batch_size, k)
    losses3[k] = np.mean(loss3)
    abl_acc3.append(acc_val3)
    abl_loss3.append(val_loss3.mean())

    #baseline
    loss4 = train_epoch(model4, loss_fn4, batch_size, k, opt4, "4")
    acc_train4, train_loss4 = test_train(model4, loss_fn4, batch_size, k)
    acc_val4, val_loss4 = test_val(model4, loss_fn4, batch_size, k)
    losses4[k] = np.mean(loss4)
    abl_acc4.append(acc_val4)
    abl_loss4.append(val_loss4.mean())

    # tensorboard record
    writer.add_scalars("ablation/val_loss", {"No dp": val_loss1.mean(), "No BN": val_loss2.mean(), "None": val_loss3.mean(),"Baseline":val_loss4.mean()}, k)
    writer.add_scalars("ablation/val_accuracy", {"No dp": acc_val1, "No BN": acc_val2, "None": acc_val3,"Baseline":acc_val4}, k)


# flush the records
writer.flush()
writer.close()

print("Done!")
