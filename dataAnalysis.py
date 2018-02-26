import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
originPath='E:\\Data\\kaggle\\DigitRecognizer\\'
train_data=pd.read_csv(originPath+'train.csv')
test_data=pd.read_csv(originPath+'test.csv')

'''查看训练集和测试集的简明摘要'''
train_data.info()
print('_____________________________________________________')
test_data.info()
print('_____________________________________________________')

'''查看是否存在缺失值（若存在缺失值，需要对缺失值进行处理）'''
# isnull()判断是否有缺失值；
# any()返回是否有任何元素在请求轴上为真（会将DataFrame转为Series）,若axis=0则以columns为单位，若axis=1则以index为单位,axis默认为0；
# describe()对数据进行描述性统计（对象属性会返回count计数和，unique不重复的值的数量，top最常见的值的value，freq最常见的值的频率）
print(train_data.isnull().any().describe())
print('_____________________________________________________')
print(test_data.isnull().any().describe())
print('_____________________________________________________')

'''拆分训练集的特征X和标签Y'''
X_train=train_data.drop(columns=['label'])
Y_train=train_data.label
del train_data

'''查看训练集标签Y的基本情况'''
# 绘制计数直方图
sns.countplot(Y_train)
plt.show()
# 使用pd.Series.value_counts()
print(Y_train.value_counts())


