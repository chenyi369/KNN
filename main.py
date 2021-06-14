# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#读取数据
def read_xlsx(csv_path):
    data = pd.read_csv(csv_path)
    print(data)
    return data

#归一化
def MinMaxScaler(data):
    col = data.shape[1]
    for i in range(0, col-1):
        arr = data.iloc[:, i]
        arr = np.array(arr)
        min = np.min(arr)
        max = np.max(arr)
        arr = (arr-min)/(max-min)
        data.iloc[:, i] = arr
    return data

def train_test_split(data, test_size=0.2, random_state=None):
    col = data.shape[1]
    x = data.iloc[:, 0:col-1]
    y = data.iloc[:, -1]
    x = np.array(x)
    y = np.array(y)
    # 设置随机种子，当随机种子非空时，将锁定随机数
    if random_state:
        np.random.seed(random_state)
        # 将样本集的索引值进行随机打乱
        # permutation随机生成0-len(data)随机序列
    shuffle_indexs = np.random.permutation(len(x))
    # 提取位于样本集中20%的那个索引值
    test_size = int(len(x) * test_size)
    # 将随机打乱的20%的索引值赋值给测试索引
    test_indexs = shuffle_indexs[:test_size]
    # 将随机打乱的80%的索引值赋值给训练索引
    train_indexs = shuffle_indexs[test_size:]
    # 根据索引提取训练集和测试集
    x_train = x[train_indexs]
    y_train = y[train_indexs]
    x_test = x[test_indexs]
    y_test = y[test_indexs]
    # 将切分好的数据集返回出去
    # print(y_train)
    return x_train, x_test, y_train, y_test

def CountDistance(train,test,length):
    distance = 0
    for x in range(length):
        distance += pow(test[x] - train[x], 2)**0.5
    return distance

def getNeighbor(x_train,test,y_train,k):
    distance = []
    #测试集的维度
    length = x_train.shape[1]
    #测试集合所有训练集的距离
    for x in range(x_train.shape[0]):
        dist = CountDistance(test, x_train[x], length)
        distance.append(dist)
    distance = np.array(distance)
    #排序
    distanceSort = distance.argsort()
    # distance.sort(key= operator.itemgetter(1))
    # print(len(distance))
    # print(distanceSort[0])
    neighbors =[]
    for x in range(k):
        labels = y_train[distanceSort[x]]
        neighbors.append(labels)
        # print(labels)
    counts = np.bincount(neighbors)
    label = np.argmax(counts)
    # print(label)
    return label

def getAccuracy(x_test,x_train,y_train,y_test):
    result = []
    k = 3
    arr_label = getNeighbor(x_train, x_test[0], y_train, k)
    for x in range(len(x_test)):
        arr_label = getNeighbor(x_train, x_test[x], y_train, k)
        result.append(arr_label)
    correct = 0
    for x in range(len(y_test)):
        if result[x] == y_test[x]:
           correct += 1
    # print(correct)
    accuracy = (correct / float(len(y_test))) * 100.0
    print("Accuracy:", accuracy, "%")
    return accuracy







if __name__ == '__main__':
    data = read_xlsx(r'D:\数据集\data\win.csv')
    scaler_data = MinMaxScaler(data)
    x_train,x_test,y_train,y_test = train_test_split(scaler_data)
    # getNeighbor(x_train,x_test[0],y_train,3)
    getAccuracy(x_test, x_train,y_train,y_test)




