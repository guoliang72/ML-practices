#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import linear_model

def load_data(file_name):
	'''导入训练数据
	input:  file_name(string)训练数据的位置
	output: feature_data(mat)特征
			label_data(mat)标签
	'''
	f = open(file_name)  # 打开文件
	feature_data = []
	label_data = []
	for line in f.readlines():
		feature_tmp = []
		lable_tmp = []
		lines = line.strip().split("\t")
		feature_tmp.append(1)  # 偏置项
		for i in range(len(lines) - 1):
			feature_tmp.append(float(lines[i]))
		lable_tmp.append(float(lines[-1]))
		feature_data.append(feature_tmp)
		label_data.append(lable_tmp)
	f.close()  # 关闭文件
	return np.mat(feature_data), np.mat(label_data)

def sig(x):
	'''Sigmoid函数
		input:  x(mat):feature * w
		output: sigmoid(x)(mat):Sigmoid值
	'''
	return 1.0 / (1 + np.exp(-x))

def lr_train_bgd(feature, label, maxCycle, alpha):
	'''利用梯度下降法训练LR模型
	input:  feature(mat)特征
			label(mat)标签
			maxCycle(int)最大迭代次数
			alpha(float)学习率
	output: w(mat):权重
	'''
	n = np.shape(feature)[1]  # 特征个数
	w = np.mat(np.ones((n, 1)))  # 初始化权重
	i = 0
	while i <= maxCycle:  # 在最大迭代次数的范围内
		i += 1  # 当前的迭代次数
		h = sig(feature * w)  # 计算Sigmoid值
		err = label - h
		w = w + alpha * feature.T * err  # 权重修正
	return w

def test_LogisticRegression(feature,label):
	regr = linear_model.LogisticRegression(solver='liblinear')
	regr.fit(feature,label)
	print('Coefficients:%s, intercept %s' % (regr.coef_, regr.intercept_))
	return regr.coef_

if __name__ == "__main__":
	# 1、导入训练数据
	print("---------- 1.load data ------------")
	feature, label = load_data("testData/data.txt")
	# 2、训练LR模型
	print("---------- 2.training ------------")
	w_lr = lr_train_bgd(feature, label, 1000000, 0.01)
	# 2、SKlearn训练LR模型
	w_sk_lr = test_LogisticRegression(feature,label)

