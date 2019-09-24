import numpy as np
from sklearn.ensemble import RandomForestClassifier
from numpy import savetxt, loadtxt

#导入训练数据集
train = loadtxt('train1.csv', delimiter=',', skiprows=1)
X_train = np.array([x[1:] for x in train])
print(X_train.shape)
Y_train = np.array([x[0] for x in train])
print(Y_train.shape)

#导入测试数据集
X_test = loadtxt('test1.csv', delimiter=',', skiprows=1)
print(X_test.shape)
print('Training...')

#初始化随机森林分类器
rf = RandomForestClassifier(n_estimators=100)
print('Predicting...')

#生成训练模型
rf_model = rf.fit(X_train, Y_train)

#利用生成的模型对数据进行预测
predictResult = rf_model.predict(X_test)

#将预测结果添加序号
pred = [[index + 1, x] for index, x in enumerate(predictResult)]

#将结果写入CSV文件
savetxt('mypredict1.csv', pred, delimiter=',', fmt='%d,%d', header='ImageId,Label', comments='')
print('Done.' )