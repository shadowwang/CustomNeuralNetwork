import numpy as np
import scipy.special as sc
import matplotlib.pyplot as mp

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

    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activate_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activate_function(final_inputs)

        targets = np.array(target_list, ndmin=2).T

        # 输出层的误差
        output_errors = targets - final_outputs
        self.who += self.learnGrate * np.dot(output_errors * final_outputs * (1 - final_outputs),
                                             np.transpose(hidden_outputs))
        # 隐藏层的误差
        hidden_errors = np.dot(self.who.T, output_errors)
        self.wih += self.learnGrate * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs),
                                             np.transpose(inputs))





inputNodes = 784# 28*28的输入矩阵
hiddenNodes = 100
outputNodes = 10# 十个输出节点，对应数字0-9
learnGrate = 0.5

n = neturalNetWork(inputNodes, hiddenNodes, outputNodes, learnGrate)
# print(n.query([1.0, -0.5, 1.5]))

# 训练集
data_file = open("dataset/mnist_train_100.csv", 'r')
data_list = data_file.readlines()
data_file.close()

for record in data_list:
    all_values = record.split(",")
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(outputNodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

# 测试集
test_data_file = open("dataset/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scoreCard = []

for test_record in test_data_list:
    all_test_values = test_record.split(",")
    result_digit = int(all_test_values[0])
    print("正确的手写数字结果是：", result_digit)

    input_list = (np.asfarray(all_test_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(input_list)
    label = np.argmax(outputs)
    print("神经网络的识别结果是：", label)
    print("####################")




