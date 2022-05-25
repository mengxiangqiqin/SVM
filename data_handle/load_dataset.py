import os
import cv2
import random
import numpy as np

def loadImages(dirName):
	filelist = os.listdir(dirName)
	random.shuffle(filelist)

	datalist = []
	labellist = []
	for data_path in filelist:
		# 添加label标签，银杏树设为1， 其他树设为0

		if data_path[0:6] == 'ginkgo':
			label = random.randint(0,6)
			labellist.append(label)
		else:
			label = random.randint(5, 11)
			labellist.append(label)

		# 将图像数据转为灰度图
		data_path = dirName + '/' + data_path
		data = cv2.imread(data_path)
		gray_data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

		#二值化处理
		# ret, thresh1 = cv2.threshold(gray_data, 127, 255, cv2.THRESH_BINARY)

		# 调整大小，拉伸为一维
		gray_data = cv2.resize(gray_data, (320, 320))
		gray_data = gray_data.reshape(-1)

		# 加入列表
		datalist.append(gray_data)

	# 转为数组
	dataMat = np.array(datalist)
	return dataMat, labellist