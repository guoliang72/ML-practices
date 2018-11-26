#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
from sklearn import linear_model
def load_data(file_path):
    '''导入数据
    input:  file_path(string):训练数据
    output: feature(mat):特征
            label(mat):标签
    '''
    f = open(file_path)
    feature = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        feature_tmp.append(1)  # x0
        for i in range(0,len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)
        label.append(float(lines[-1]))
    f.close()
    return np.mat(feature), np.mat(label).T

def ridge_regression(feature,label,lam):
	n = np.shape(feature)[1]
	w = (feature.T * feature + lam * np.mat(np.eye(n))).I * feature.T * label
	return w

def ridge_GD(feature, label, maxCycle, alpha,lam):
	n = np.shape(feature)[1]
	m = np.shape(feature)[0]
	w = np.mat(np.ones((n, 1)))
	i = 0
	while i <= maxCycle:
		i += 1
		error = feature * w - label
		w = w - alpha * (feature.T * error + lam * np.mat(np.eye(n)) * w)/m
	return w
def test_RidgeRegression(feature,label):
	regr = linear_model.Ridge()
	regr.fit(feature,label)
	print('Coefficients:%s, intercept %.2f' % (regr.coef_, regr.intercept_))
	return regr.coef_

if __name__ == "__main__":
    # 1、导入数据集
    print("\t----------- 1.load data ----------")
    feature, label = load_data("testData/lineardata.txt")
    # 2.1、最小二乘求解
    print("\t----------- 2.training ----------")
    print("\t ---------- least_square ----------")
    w_ridge = ridge_regression(feature, label,0.5)
    # 2.2、SkLearn内置方法
    print("\t ---------- newton ----------")
    w_sklearn = test_RidgeRegression(feature, label)
    # 2.3、梯度下降
    print("\t ---------- newton ----------")
    w_GD = ridge_GD(feature, label,10000,0.01,0.5)
