import numpy as np


class Perceptron:

    def __init__(self, parameters1, result1, parameters2, result2):
        self.x1 = parameters1
        self.y1 = result1
        self.x2 = parameters2
        self.y2 = result2

    def onlinePerceptron(self, maxIter=1):
        weight = np.zeros((1, self.x1.T.shape[0]))
        tt = 0

        for i in range(0, maxIter):
            tt = 0
            for t in range(0, self.y1.shape[0]):

                u = np.sign(np.dot(weight, self.x1[t].T))

                if (self.y1[t]*u <= 0):
                    weight = weight + self.y1[t]*self.x1[t]
                    tt += 1
            val_t = np.dot(self.x2, weight.T)
            ct = 0
            for i in range(0, len(val_t)):
				if self.y2[i]*np.sign(val_t[i]) > 0:
					ct += 1
            val = ct/self.y2.shape[0]
            print((1-(tt/self.y1.shape[0])), val)

        return weight

    def avgPerceptron(self, maxIter=1):
        # (1*N)
        weight = np.zeros((1, self.x1.T.shape[0]))
        avgWeight = np.zeros((1, self.x1.T.shape[0]))
        count, countSum = 0.0, 0.0
        tt = 0

        for i in range(0, maxIter):

            tt = 0
            for t in range(0, self.y1.shape[0]):

                #(1*N) (N*1)
                u = np.sign(np.dot(weight, self.x1[t].T))

                if (self.y1[t]*u <= 0):
                    if (count + countSum) > 0:
						a = countSum / (count + countSum)
						b = count / (count + countSum)
						avgWeight=  a*avgWeight + b*weight
                    countSum += count
                    weight = weight + self.y1[t]*self.x1[t]
                    count = 0
                    tt += 1
                else:
                    count += 1
            val_t = np.dot(self.x2, avgWeight.T)
            ct = 0
            for i in range(0, len(val_t)):
				if self.y2[i]*np.sign(val_t[i]) > 0:
					ct += 1
            val = ct/self.y2.shape[0]
            print((1-(tt/self.y1.shape[0])), val)
        if count > 0:
			a = countSum / (count + countSum)
			b = count / (count + countSum)
			avgWeight =  a*avgWeight + b*weight
        return avgWeight
