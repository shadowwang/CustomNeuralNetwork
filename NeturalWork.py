import numpy as np
import scipy.special as sc

# 自定义神经网络
class neturalNetWork:

    def __init__(self, inputNodes,
                 hiddenNodes, outputNodes,
                 learnGrate):
        # 输入层节点数
        self.inputNodes = inputNodes
        # 隐藏层节点数
        self.hiddenNodes = hiddenNodes
        # 输出层节点数
        self.outputNodes = outputNodes
        # 学习率
        self.learnGrate = learnGrate

        # 输入层到隐藏层权重矩阵
        self.wih = np.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.inputNodes))
        # 隐藏层到输出层权重矩阵
        self.who = np.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes, self.hiddenNodes))

        # 激活S函数定义
        self.activate_function = lambda x: sc.expit(x)

    def query(self, input_list):

        inputs = np.array(input_list, ndmin=2).T
        # 隐藏层的输入=输入层权重*输入数据
        hidden_inputs = np.dot(self.wih, inputs)
        # 隐藏层的输出 = 激活函数(隐藏层的输入)
        hidden_outputs = self.activate_function(hidden_inputs)

        # 输出层的输入 = 隐藏层权重 * 隐藏层输出
        final_inputs = np.dot(self.who, hidden_outputs)
        # 输出层的输出 = 激活函数（输出层的输入）
        final_outputs = self.activate_function(final_inputs)

        return final_outputs

    def train(self):
        pass

inputNodes = 3
hiddenNodes = 3
outputNodes = 3
learnGrate = 0.5

n = neturalNetWork(inputNodes, hiddenNodes, outputNodes, learnGrate)
print(n.query([1.0, -0.5, 1.5]))

