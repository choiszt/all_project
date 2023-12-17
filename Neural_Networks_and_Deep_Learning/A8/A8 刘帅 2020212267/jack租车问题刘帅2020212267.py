from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import poisson
import seaborn as sns

parkingnum = 20 #每个租车点最多可以停放20辆车
lamda1_rent = 3 #停车场1租车λ值（3）
lamda1_return = 3 #停车场1还车λ值（3）
lamda2_rent = 4 #停车场2租车λ值（4）
lamda2_return = 2 #停车场2还车λ值（2）
MAX_ACTION = 5 #最大调配汽车数目（5）
DISCOUNT = 0.09 #收益折扣
cost = 2 #每调配一辆车花费2美金。
earn = 10 #租车的收入（10）
actions = np.arange(-MAX_ACTION, MAX_ACTION + 1) #动作集合（-5，-4，…，4，5）
poisson_max = 11 #限制泊松分布产生请求数目的上限
poisson_cache = dict() # 存储每个（n，λ）对应的泊松概率，key为n*(poisson_max-1)+lam


def poisson_prob(n, lam):
    global poisson_cache
    key = n * (poisson_max - 1) + lam
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]


class dp:
    def __init__(self):
        self.v = np.ones((parkingnum + 1, parkingnum + 1), float)
        self.actions = np.zeros((parkingnum + 1, parkingnum + 1), int)
        self.gamma = DISCOUNT
        self.delta = 0
        self.theta = 0.01
        pass

    def state_value(self, state, action, state_value):
        """
        :state: 状态定义为每个地点的车辆数
        :action: 车辆的移动数量[-5,5]，负：2->1，正：1->2
        :state_value: 状态价值矩阵
        """
        # initial total return
        returns = 0.0

        # 移动车辆产生负收益
        returns -= cost * abs(action)

        # 移动后的车辆总数不能超过20
        NUM_OF_CARS_1 = min(state[0] - action, parkingnum)
        NUM_OF_CARS_2 = min(state[1] + action, parkingnum)

        # 遍历两地全部的可能概率下（截断泊松概率）租车请求数目
        for rent_1 in range(poisson_max):
            for rent_2 in range(poisson_max):
                # prob为两地租车请求的联合概率，概率为泊松分布
                prob = poisson_prob(rent_1, lamda1_rent) * poisson_prob(rent_2, lamda2_rent)
                # 两地原本汽车数量
                num_of_cars_1 = NUM_OF_CARS_1
                num_of_cars_2 = NUM_OF_CARS_2
                # 有效租车数目必须小于等于该地原有的车辆数目
                valid_rent_1 = min(num_of_cars_1, rent_1)
                valid_rent_2 = min(num_of_cars_2, rent_2)
                # 计算回报，更新两地车辆数目变动
                reward = (valid_rent_1 + valid_rent_2) * earn
                num_of_cars_1 -= valid_rent_1
                num_of_cars_2 -= valid_rent_2
                # 如果还车数目为泊松分布的均值
                    # 两地的还车数目均为泊松分布均值
                returned_cars_1 = lamda1_return
                returned_cars_2 = lamda2_return
                # 还车后总数不能超过车场容量
                num_of_cars_first_loc = min(num_of_cars_1 + returned_cars_1, parkingnum)
                num_of_cars_second_loc = min(num_of_cars_2 + returned_cars_2, parkingnum)
                # 核心：
                # 策略评估：V(s) = p(s',r|s,π(s))[r + γV(s')]
                returns += prob * (reward + DISCOUNT * state_value[num_of_cars_first_loc, num_of_cars_second_loc])

                # 否则计算所有泊松概率分布下的还车空间

        return returns

    def policy_iteration(self):
        # 设置迭代参数
        iterations = 0
        # 准备画布大小，并准备多个子图
        _, axes = plt.subplots(2, 2, figsize=(40, 20))
        # 调整子图的间距，wspace=0.1为水平间距，hspace=0.2为垂直间距
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        # 这里将子图形成一个1*4的列表
        axes = axes.flatten()
        while True:
            # 使用seaborn的heatmap作图
            fig = sns.heatmap(np.flipud(self.actions), cmap="rainbow", ax=axes[iterations])
            # 定义标签与标题
            fig.set_ylabel('# cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(parkingnum + 1))))
            fig.set_xlabel('# cars at second location', fontsize=30)
            fig.set_title('policy {}'.format(iterations), fontsize=30)
            # policy evaluation (in-place) 策略评估（in-place）
            # 未改进前，第一轮policy全为0，即[0，0，0...]
            while True:
                old_value = self.v.copy()
                for i in range(parkingnum + 1):
                    for j in range(parkingnum + 1):
                        # 更新V（s）
                        new_state_value = self.state_value([i, j], self.actions[i, j], self.v)
                        # in-place操作
                        self.v[i, j] = new_state_value
                # 比较V_old(s)、V(s)，收敛后退出循环
                max_value_change = abs(old_value - self.v).max()
                print('max value change {}'.format(max_value_change))
                if max_value_change < 1e-4:
                    break

            # policy improvement
            # 在上一部分可以看到，策略policy全都是0，如不进行策略改进，其必然不会收敛到实际最优策略。
            # 所以需要如下策略改进
            policy_stable = True
            # i、j分别为两地现有车辆总数
            for i in range(parkingnum + 1):
                for j in range(parkingnum + 1):
                    old_action = self.actions[i, j]
                    action_returns = []
                    # actions为全部的动作空间，即[-5、-4...4、5]
                    for action in actions:
                        if (0 <= action <= i) or (-j <= action <= 0):
                            action_returns.append(self.state_value([i, j], action, self.v))
                        else:
                            action_returns.append(-np.inf)
                    # 找出产生最大动作价值的动作
                    new_action = actions[np.argmax(action_returns)]
                    # 更新策略
                    self.actions[i, j] = new_action
                    if policy_stable and old_action != new_action:
                        policy_stable = False
            print('policy stable {}'.format(policy_stable))

            if policy_stable:
                fig = sns.heatmap(np.flipud(self.v), cmap="rainbow", ax=axes[-1])
                fig.set_ylabel('# cars at first location', fontsize=30)
                fig.set_yticks(list(reversed(range(parkingnum + 1))))
                fig.set_xlabel('# cars at second location', fontsize=30)
                fig.set_title('optimal value', fontsize=30)
                break

            iterations += 1

        plt.savefig('./policy_iteration.png')
        plt.show()
        plt.close()
        return

    def value_iteration(self):
        # 设置迭代参数
        iterations = 0

        # 准备画布大小，并准备多个子图
        _, axes = plt.subplots(2, 2, figsize=(40, 20))
        # 调整子图的间距，wspace=0.1为水平间距，hspace=0.2为垂直间距
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        # 这里将子图形成一个1*4的列表
        axes = axes.flatten()
        while True:
            # 使用seaborn的heatmap作图
            fig = sns.heatmap(np.flipud(self.actions), cmap="rainbow", ax=axes[iterations])

            # 定义标签与标题
            fig.set_ylabel('# cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(parkingnum + 1))))
            fig.set_xlabel('# cars at second location', fontsize=30)
            fig.set_title('policy {}'.format(iterations), fontsize=30)

            # value iteration 价值迭代
            while True:
                old_value = self.v.copy()
                for i in range(parkingnum + 1):
                    for j in range(parkingnum + 1):
                        action_returns = []
                        # actions为全部的动作空间，即[-5、-4...4、5]
                        for action in actions:
                            if (0 <= action <= i) or (-j <= action <= 0):
                                action_returns.append(self.state_value([i, j], action, self.v))
                            else:
                                action_returns.append(-np.inf)
                        # 找出产生最大动作价值的动作
                        max_action = actions[np.argmax(action_returns)]
                        # 更新V（s）
                        new_state_value = self.state_value([i, j], max_action, self.v)
                        # in-place操作
                        self.v[i, j] = new_state_value
                # 比较V_old(s)、V(s)，收敛后退出循环
                max_value_change = abs(old_value - self.v).max()
                print('max value change {}'.format(max_value_change))
                if max_value_change < 1e-4:
                    break

            # policy improvement
            policy_stable = True
            # i、j分别为两地现有车辆总数
            for i in range(parkingnum + 1):
                for j in range(parkingnum + 1):
                    old_action = self.actions[i, j]
                    action_returns = []
                    # actions为全部的动作空间，即[-5、-4...4、5]
                    for action in actions:
                        if (0 <= action <= i) or (-j <= action <= 0):
                            action_returns.append(self.state_value([i, j], action, self.v))
                        else:
                            action_returns.append(-np.inf)
                    # 找出产生最大动作价值的动作
                    new_action = actions[np.argmax(action_returns)]
                    # 更新策略
                    self.actions[i, j] = new_action
                    if policy_stable and old_action != new_action:
                        policy_stable = False
            print('policy stable {}'.format(policy_stable))

            if policy_stable:
                fig = sns.heatmap(np.flipud(self.v), cmap="rainbow", ax=axes[-1])
                fig.set_ylabel('# cars at first location', fontsize=30)
                fig.set_yticks(list(reversed(range(parkingnum + 1))))
                fig.set_xlabel('# cars at second location', fontsize=30)
                fig.set_title('optimal value', fontsize=30)
                break

            iterations += 1

        plt.savefig('./value_iteration.png')
        plt.show()
        plt.close()
        return


if __name__ == '__main__':
    model = dp()

    #model.policy_iteration()
    model.value_iteration()
    print(poisson_cache)
    pass