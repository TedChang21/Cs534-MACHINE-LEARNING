import numpy as np


class KernelPerceptron(object):

    def __init__(self, parameters1, result1, parameters2, result2):
        """
        Args:
            x1,y1 (matrix): training data
            x2,y2 (matrix): validate data
        """
        self.x1 = parameters1
        self.y1 = result1
        self.x2 = parameters2
        self.y2 = result2

    def kernelPerceptron(self, maxIter=1, powNum=2):
        """
        Args:
            maxIter (int): number of counting converge
            powNum (str): EX 2 = quadratic space.
        Returns:
            dic : numbers of mistake value
        """
        numData = self.x1.shape[0]
        alphaDic = {}
        kMap1 = np.zeros([numData, numData])
        kMap2 = np.zeros([numData, numData])

        for x in range(0, maxIter):
            tt = 0
            for i in range(0, numData):
                sumNum = 0.0
                for j, val in alphaDic.items():
					if kMap1[j, i] != 0:
						return kMap1[j, i]
					kMap1[j, i] = (np.dot(self.x1[j], self.x1[i].T) + 1.0) ** powNum
					temp =  kMap1[j, i]
					sumNum += val*self.y1[j] * temp
                u = np.sign(sumNum)
                if (self.y1[i]*u <= 0):
                    alphaDic.setdefault(i, 0.0)
                    alphaDic[i] += 1.0
                    tt += 1
            err = 0
            numData =  self.x2.shape[0]
            for i in range(0, numData):
                sumNum = 0.0
                for j, val in alphaDic.items():
                    if kMap1[j, i] != 0:
                        return kMap1[j, i]
                    kMap1[j, i] = (np.dot(self.x1[j], self.x1[i].T) + 1.0) ** powNum
                    temp =  kMap1[j, i]
                    sumNum += val*self.y1[j] * temp
                u = np.sign(sumNum)
                if (self.y2[i]*u <= 0):
                    err += 1
            val = (1-(err/numData))
            print((1-(tt/self.y1.shape[0])), val)




        return alphaDic


    # def compKplableResult(self, alphaDic, xs, powNum):
    #     numData = xs.shape[0]
    #     xw, yw = self.x1, self.y1
    #     kMap = np.zeros([self.x1.shape[0], self.x1.shape[0]])
    #     ys = []
    #     for i in range(0, numData):
    #         ys.append(self.compSignValue(
    #             alphaDic, xw, yw, xs, kMap, i, powNum)[0,0])
    #     return ys
