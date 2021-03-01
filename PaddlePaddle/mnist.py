import numpy as np
import paddle as paddle
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt
import os
train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.mnist.train(),
                                                  buf_size=512),
                    batch_size=128)
test_reader = paddle.batch(paddle.dataset.mnist.test(),
                           batch_size=128)
# 定义多层感知器
def multilayer_perceptron(input):
    # 第一个全连接层，激活函数为ReLU
    hidden1 = fluid.layers.fc(input=input, size=100, act='relu')
    # 第二个全连接层，激活函数为ReLU
    hidden2 = fluid.layers.fc(input=hidden1, size=100, act='relu')
    # 以softmax为激活函数的全连接输出层，大小为10
    prediction = fluid.layers.fc(input=hidden2, size=10, act='softmax')
    return prediction
# 定义输入输出层
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')  #单通道，28*28像素值
label = fluid.layers.data(name='label', shape=[1], dtype='int64')            #图片标签
# 获取分类器
model = multilayer_perceptron(image)
# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)  #使用交叉熵损失函数,描述真实样本标签和预测概率之间的差值
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)
# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)   #使用Adam算法进行优化
opts = optimizer.minimize(avg_cost)
# 定义一个使用CPU的解析器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())
# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
# 开始训练和测试
for pass_id in range(5):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):  # 遍历train_reader
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),  # 运行主程序
                                        feed=feeder.feed(data),  # 给模型喂入数据
                                        fetch_list=[avg_cost, acc])  # fetch 误差、准确率
        # 每100个batch打印一次信息  误差、准确率
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 进行测试
    test_accs = []
    test_costs = []
    # 每训练一轮 进行一次测试
    for batch_id, data in enumerate(test_reader()):  # 遍历test_reader
        test_cost, test_acc = exe.run(program=fluid.default_main_program(),  # 执行训练程序
                                      feed=feeder.feed(data),  # 喂入数据
                                      fetch_list=[avg_cost, acc])  # fetch 误差、准确率
        test_accs.append(test_acc[0])  # 每个batch的准确率
        test_costs.append(test_cost[0])  # 每个batch的误差
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))  # 每轮的平均误差
    test_acc = (sum(test_accs) / len(test_accs))  # 每轮的平均准确率
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))

    # 保存模型
    model_save_dir = "../home/aistudio/data/hand.inference.model"
    # 如果保存路径不存在就创建
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print('save models to %s' % (model_save_dir))
    fluid.io.save_inference_model(model_save_dir,  # 保存推理model的路径
                                  ['image'],  # 推理（inference）需要 feed 的数据
                                  [model],  # 保存推理（inference）结果的 Variables
                                  exe)  # executor 保存 inference model
