# tensorflow_classifier

1、datasets文件夹下的代码是对各种数据的预处理操作，产生一批tfrecord文件（最好使用这种方法，所有数据都放在一个tfrecord文件的话，一次加载数据过大造成内存溢出。）
2、nets文件夹下面存放的是各种各样的分类网络（inception_1，resnet，vgg等，后续会更新）
3、preprocessing文件夹下面的代码是对不同的网络进行不同的图像预处理
4、train.py：训练文件
5、predict.py：验证（预测）文件，后续更新
