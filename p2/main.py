import numpy as np
import datetime as dt
import perceptron as pct
import kernelPerceptron as kpt
import fileHelper as fh

def importCsv(path, isSplit=True):

    datas = np.genfromtxt(path, delimiter=',')
    if isSplit:
        return splitDataSet(datas)
    else:
        return np.c_[np.ones(datas.shape[0]), datas]


def splitDataSet(datas):
    parameters = datas[:, 1:datas.shape[1]]
    result = datas[:, 0:1]
    # Insert Dummy Value to first column
    parameters = np.c_[np.ones(datas.shape[0]), parameters]
    return (parameters, result)


def setLabels(arr, posVal, negVal):
    arr[arr == posVal] = 1
    arr[arr == negVal] = -1


def predictValue(x, wgt): 
    return np.sign(np.dot(x, wgt.T))

# =============================================================================
# ################ Main Function ################
# =============================================================================
maxIter = 15                                # Online\Avg Perceptron Loop number
kerIter = 1                                 # Kernel Perceptron Loop number
powNum = 2                                  # Kernel function power number
fileName1 = "pa2_train.csv"                 # Training File name
fileName2 = "pa2_valid.csv"                 # Validate File name
fileName3 = "pa2_test_no_label.csv"         # Validate File name

print("\n ------------ ImportDaTa ------------")
(par1, rst1) = importCsv(fileName1)
(par2, rst2) = importCsv(fileName2)
setLabels(rst1, 3, 5)
setLabels(rst2, 3, 5)
par1 = np.matrix(par1)
rst1 = np.matrix(rst1)
par2 = np.matrix(par2)
rst2 = np.matrix(rst2)

print("\n ------------ Perceptron ------------")
pt = pct.Perceptron(par1, rst1, par2, rst2)
w = pt.onlinePerceptron(maxIter)

print("\n ------------ AvgPerceptron ------------")
pt = pct.Perceptron(par1, rst1, par2, rst2)
w = pt.avgPerceptron(maxIter)

print("\n ------------ kerPerceptron ------------")
print(dt.datetime.now())
kp = kpt.KernelPerceptron(par1, rst1, par2, rst2)
kp.kernelPerceptron(kerIter, powNum)
# print("\n ------------ kerPerceptron ------------")
# kp.kernelPerceptron(15,2)
# print("\n ------------ kerPerceptron ------------")
# kp.kernelPerceptron(15,3)
# print("\n ------------ kerPerceptron ------------")
# kp.kernelPerceptron(15,7)
# print("\n ------------ kerPerceptron ------------")
# kp.kernelPerceptron(15,15)
print(dt.datetime.now())

# par3 = hp.importCsv(fileName3, False)
# result = hp.predictValue(par3, w)
# fOut = fh.fileHelper("oplabel2.csv")
# fOut.outputResult(result)
# fOut.fileClose()

# alphaDic = kp.kernelPerceptron(4, 3)
# print(dt.datetime.now())
# xs = hp.importCsv(fileName3, False)
# print(xs.shape)
# result = kp.compKplableResult(alphaDic, xs, 3)
# fOut = fh.fileHelper("kplable.csv")
# fOut.outputResult(result)
# fOut.fileClose()
