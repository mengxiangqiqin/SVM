import numpy as np
from smo import smoP,kernelTrans
from data_handle.load_dataset import loadImages


def testDigits(kTup=('rbf', 10)):
	"""
	测试函数
	Parameters:
		kTup - 包含核函数信息的元组
	Returns:
	    无
	"""
	print('==========================正在加载数据集==================================')
	dataArr,labelArr = loadImages(r'E:\ML\second_work\SVM\ginkgo\train')
	print('==========================加载数据集已完成==================================')

	print('======================开始计算smoP========================================')
	b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10, kTup)
	print('=====================smoP计算已完成============================================')
	datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
	svInd = np.nonzero(alphas.A>0)[0]
	sVs=datMat[svInd]
	labelSV = labelMat[svInd];
	print("支持向量个数:%d" % np.shape(sVs)[0])
	m,n = np.shape(datMat)


	errorCount = 0
	print('=======================开始进行迭代===============================================')
	for i in range(m):
		kernelEval = kernelTrans(sVs,datMat[i,:],kTup)

		predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
		if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
	print('======================================训练集结果输出=========================================')
	print("训练集错误率: %.2f%%" % (float(errorCount)/m*100))
	print('训练集正确率:')

	dataArr,labelArr = loadImages(r'E:\ML\second_work\SVM\ginkgo\test')
	errorCount = 0
	datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
	m,n = np.shape(datMat)
	for i in range(m):
		kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
		predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b

		if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1


	print("测试集错误率: %.2f%%" % (float(errorCount)/m*100))

if __name__ == '__main__':
	testDigits()