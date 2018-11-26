#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import neighbors

def KNN(feature,label,test_data,k=3):
	n = np.shape(feature)[0]
	expend_data = np.tile(test_data,(n,1))
	dis = np.power((feature - expend_data),2)
	distance = list(np.power(dis.sum(axis=1),0.5))
	dis_sort = sorted(distance)
	# 找到前K个
	k_dis = dis_sort[:7]
	# 在dis中找到对应的前K个元素的下标
	k_index = []
	for each_dis in k_dis:
		k_index.append(distance.index(each_dis))
	k_labels = label[k_index]
	print(k_labels)
	# 计算labels中统计次数
	from collections import Counter
	c = Counter(k_labels).most_common()
	print(c)
	print("test_data should to be judge to be :", c[0][0])

def test_KNeighborsClassifier(feature,label,test_data):
	feature = np.mat(feature)
	label = np.mat(label).T
	test_data = np.mat(test_data)
	knn=neighbors.KNeighborsClassifier(n_neighbors=3)
	knn.fit(feature,label)
	print("Training Score:%f"%knn.score(feature,label))
	print("test_data should to be judge to be :",knn.predict(test_data))


if __name__ == '__main__':
	import numpy as np
	data = np.array([[3, 104, 0], [2, 100, 0], [1, 81, 0], [101, 10, 1], [99, 5, 1], [98, 2, 1]])
	feature = data[:,:-1]
	label = data[::,-1]
	test_data = np.array([90, 3])
	KNN(feature,label, test_data, 3)
	test_KNeighborsClassifier(feature,label,test_data)