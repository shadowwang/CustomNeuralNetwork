
class neturalNetWork:


    def __init__(self, inputNodes,
                 hiddenNodes, outputNodes,
                 learnGrate):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.learnGrate = learnGrate

    def train(self):
        pass

    def query(self):
        pass

inputNodes = 3
hiddenNodes = 3
outputNodes = 3
learnGrate = 0.5

n = neturalNetWork(inputNodes, hiddenNodes, outputNodes, learnGrate)

