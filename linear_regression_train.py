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
def least_square(feature,label):
    w = (feature.T * feature).I * feature.T * label
    return w


def get_error(feature,label,w):
    return (label - feature * w).T * (label - feature * w)/2

def test_LinearRegression(feature, label):
    regr = linear_model.LinearRegression()
    regr.fit(feature, label)
    print('Coefficients:%s, intercept %.2f'%(regr.coef_,regr.intercept_))
    return regr.coef_

if __name__ == "__main__":
    # 1、导入数据集
    print("----------- 1.load data ----------")
    feature, label = load_data("testData/lineardata.txt")
    # 2.1、最小二乘求解
    print("----------- 2.training ----------")
    print("\t ---------- least_square ----------")
    w_ls = least_square(feature, label)
    # 2.2、SkLearn内置方法
    print("\t ---------- newton ----------")
    w_sklearn = test_LinearRegression(feature, label)
    # 3、保存最终的结果
    print("----------- 3.save result ----------")
    #save_model("weights", w_newton)