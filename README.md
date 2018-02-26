# Kaggle_DigitRecognizer


# 介绍

这个比赛是Kaggle上的一个[手写数字识别比赛](https://www.kaggle.com/c/digit-recognizer)，主要供新手进行入门学习，所需相关数据可到[该网站](https://www.kaggle.com/c/digit-recognizer/data)下载。

下面的Demo主要是以深度学习框架Keras搭建的卷积神经网络CNN为模型，比赛成绩为Top10%，该Demo提供了比赛的基本思路，如有错漏或相关问题，欢迎提出。

源码地址为：https://github.com/Zheng-Wenkai/Kaggle_DigitRecognizer

----------


# 一、数据总览


###1.1 查看数据基本情况
```
originPath='E:\\Data\\kaggle\\DigitRecognizer\\'
train_data=pd.read_csv(originPath+'train.csv')
test_data=pd.read_csv(originPath+'test.csv')

train_data.info()
print('_____________________________________________________')
test_data.info()
```
**运行结果：**
```
RangeIndex: 42000 entries, 0 to 41999
Columns: 785 entries, label to pixel783
_____________________________________________________
RangeIndex: 28000 entries, 0 to 27999
Columns: 784 entries, pixel0 to pixel783
```
由运行结果可知，train_data有42000行，行名称由0到41999，有785列，列名称由label到pixel783。test_data同理

### 1.2 查看是否存在缺失值

1. isnull()判断是否有缺失值；
2. any()返回是否有任何元素在请求轴上为真（会将DataFrame转为Series）,若axis=0则以columns为单位，若axis=1则以index为单位,axis默认为0；
3. describe()对数据进行描述性统计（对象属性会返回count计数和，unique不重复的值的数量，top最常见的值的value，freq最常见的值的频率）

```
print(train_data.isnull().any().describe())
print('_____________________________________________________')
print(test_data.isnull().any().describe())
```

**运行结果：**

```
count       785
unique        1
top       False
freq        785
dtype: object
_____________________________________________________
count       784
unique        1
top       False
freq        784
dtype: object
```
由运行结果可知，该数据集不存在缺失值；若存在缺失值，还需要另外加以处理

###1.3 查看训练集标签Y的情况

```
X_train=train_data.drop(columns=['label'])
Y_train=train_data.label
del train_data
# 绘制计数直方图
sns.countplot(Y_train)
plt.show()
# 使用pd.Series.value_counts()
print(Y_train.value_counts())
```
**运行结果**
![这里写图片描述](http://img.blog.csdn.net/20180226111006526?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3RhcnRlcl9fX19f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

由运行结果可知，训练集数据均匀；若数据不均匀（即某一类的数据集数量多，而某一类的数据集数量少），则需要对数据进行处理（如数据扩增或增加新的训练样本）


----------
# 二、建立base_model#

### 2.1 数据预处理

```
X_train=train_data.drop(columns=['label'])
Y_train=train_data.label
del train_data
# 改变维度：第一个参数是图片数量，后三个参数是每个图片的维度
X_train = X_train.values.reshape(-1,28,28,1)
test_data = test_data.values.reshape(-1,28,28,1)
print(X_train.shape)
print(test_data.shape)
print("Train Sample:",X_train.shape[0])
print("Test Sample:",test_data.shape[0])
# 归一化：将数据进行归一化到0-1 因为图像数据最大是255
X_train=X_train/255.0
test_data=test_data/255.0
# 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵
Y_train = to_categorical(Y_train, num_classes = nb_classes)
X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.1)
plt.imshow(X_train[0][:,:,0], cmap="Greys")
plt.show()
```
**运行结果：**
![这里写图片描述](http://img.blog.csdn.net/20180226111601940?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3RhcnRlcl9fX19f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

由运行结果可知，数据处理成功将csv数据转化成图片

### 2.2 建立CNN模型

```
model = Sequential()
# filters：卷积核的数目（即输出的维度）
# kernel_size：卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
model.add(Conv2D(filters = 32, kernel_size = (5,5),activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
```
### 2.3 编译和训练模型

```
optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# 使用多类的对数损失categorical_crossentropy
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

history=model.fit(X_train,Y_train,
           batch_size=batch_size,
           epochs=nb_epoch,
           verbose=2,
           validation_data=(X_val,Y_val))
```


----------
# 三、评估base_model#

### 3.1 查看验证集的loss和accuracy
```
score = model.evaluate(X_val, Y_val, verbose=0)
print('Val loss:', score[0])
print('Val accuracy:', score[1])
```
由运行结果可知，验证集的accuracy为98.9%

### 3.2 绘制学习曲线

```
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
ax[0].legend(loc='best', shadow=True)
ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax[1].legend(loc='best', shadow=True)
plt.show()
```
**运行结果：**
![这里写图片描述](http://img.blog.csdn.net/20180226112705634?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3RhcnRlcl9fX19f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

由运行结果可知，validation 的 loss 和 accuracy 的波动较大，且traning的学习曲线情况优于validation，故base_model出现了轻微的过拟合现象。

### 3.3 查看混淆矩阵 

混淆矩阵的每一列代表了预测类别 ，每一列的总数表示预测为该类别的数据的数目；每一行代表了数据的真实归属类别，每一行的数据总数表示该类别的数据实例的数目。

```
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap="Blues"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 是否进行标准化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# 根据验证集标签的真实值和预测值计算混淆矩阵（confusion_matrix）
Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred,axis = 1)
Y_true = np.argmax(Y_val,axis = 1)
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# 绘制混淆矩阵
plot_confusion_matrix(confusion_mtx, classes = range(10))
```

**运行结果：**
![这里写图片描述](http://img.blog.csdn.net/20180226113300883?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3RhcnRlcl9fX19f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

由运行结果可知：错误分类主要集中在对7的分类（其中有8个样本被从9误分类为7），混淆矩阵可以为我们进一步优化模型提供指导。

### 3.4 查看最显著的错误

```
def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)),cmap="Greys")
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1
    plt.show()

errors = (Y_pred_classes - Y_true != 0)  # 矩阵相减得到误差集（?*1）
# 使用布尔索引
Y_pred_classes_errors = Y_pred_classes[errors]  # 误差集的预测标签Y（?*1）
Y_pred_errors = Y_pred[errors]  # 误差集的预测序列（?*10）
Y_true_errors = Y_true[errors]  # 误差集的真实标签Y（?*1）
X_val_errors = X_val[errors]  # 误差集的特征X，即数字图片（?*28*28*1）

# 误差集中对错误标签的预测概率
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)
# 误差集中对真实标签的预测概率，np.diagonal是返回对角线的元素
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
# 排序并返回的是数组值从小到大的索引值
sorted_dela_errors = np.argsort(delta_pred_true_errors)
most_important_errors = sorted_dela_errors[-6:]
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
```

**运行结果：**
![这里写图片描述](http://img.blog.csdn.net/20180226113824580?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3RhcnRlcl9fX19f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

可视化错误分类的样本（可以通过对代码的修改来调整可视化样本的数量），从而可以得知错误分类的具体情况，为我们进一步优化模型提供指导。


----------
# 四、模型的进一步优化#

### 4.1 数据扩增

```
# 用以生成一个batch的图像数据，支持实时数据提升
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
```

### 4.2 学习率自适应

当评价指标val_acc不在提升时，减少学习率

```
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
```
### 4.3 训练和编译模型
```
optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# 使用多类的对数损失categorical_crossentropy
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs =nb_epoch, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size+1,
                              callbacks=[learning_rate_reduction])
```
### 4.4 评估优化模型

评估步骤与上述相同，这里不再重复

1、学习曲线
![这里写图片描述](http://img.blog.csdn.net/20180226115828777?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3RhcnRlcl9fX19f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

Validation的学习曲线较为稳定，且过拟合现象明显下降，模型准确率由0.989上升为0.995。

2、混淆矩阵
![这里写图片描述](http://img.blog.csdn.net/20180226120030884?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3RhcnRlcl9fX19f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


3、最显著的错误
![这里写图片描述](http://img.blog.csdn.net/20180226120040331?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3RhcnRlcl9fX19f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 4.5 关于模型优化的进一步建议

 - 可以使用集成模型，具体可参考[机器学习 —— 集成学习](http://blog.csdn.net/starter_____/article/details/79334646)
 - 已尝试了bagging集成中的投票模型（通过对测试集进行数据扩增，使用同一模型对同一图片的不同变化进行多次评估），但模型准确度下降，故不贴出代码
 - 对模型进行改造：可以使用最新的CapsulesNet胶囊网络或尝试其他模型，而不是使用当前的CNN模型
 - 进一步挖掘新的特征


----------


# 五、对测试集进行预测

```
print('Begin to predict for testing data ...')
results = model.predict(test_data)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv(originPath+"submit.csv",index=False)
```


----------


# 六、在kaggle上提交预测结果

![这里写图片描述](http://img.blog.csdn.net/20180226154856192?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3RhcnRlcl9fX19f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

该模型对测试集预测的准确率达到0.99614，在kaggle上的排名为Top10%
