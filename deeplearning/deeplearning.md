# 目的
我不打算也没机会成为专家，但是我需要了解一下，因为我认为这是未来的方向，就像是英语一样，了解这些内容也有助于我以后与相关行业内的人交流，且考研复试会用到

# Neural Network and Deep Learning

ReLu function: rectified linear unit 修正线性单元 输出大于0

应用：
- 通过房屋信息预测房价 Standard NN
- 在线广告 Standard NN
- 图像识别 CNN 卷积
- 语音识别 带有时间序列 RNN
- 翻译 带有时间序列 RNN
- 自动驾驶 更复杂的混合的


结构数据与非结构数据很好理解了，结构数据就是表格那样的
非结构数据就是音频，图片，文本

数据量小时模型的大小不是主要的瓶颈

当数据量大时模型越大表现越好

## logistic 回归

用来处理二分分类的

sigmoid func

Logistic Regression cost function："E:\info\deeplearning\deeplearning.ai-andrew-ng\COURSE 1 Neural Networks and Deep Learning\Week 2\Week 2 Logistic Regration as a Neural network.pptx"第7页


计算图
前向传播输出，后向传播调整参数

链式求导是很重要的一个概念

向量化，用numpy的向量化确实是快的惊人 

多用矩阵运算来替代for loop

用数学符号先转换会让接下来的向量化公式容易一点

python广播其实就是隐式类型转换


```py
a = np.random.randn(5) # 不要用这样的函数声明对象, 因为声明出来的对象很含糊 rank 1 array
a = np.random.randn(5, 1) # 这样的好

assert(a.shape == (5, 1)) # 或用这种断言确认一下

a = a.reshape((5, 1))
```

```py
# 可以快速查看注释
import numpy as np
np.exp?
```

normalization很重要

**What you need to remember:** - np.exp(x) works for any np.array x and applies the exponential function to every coordinate - the sigmoid function and its gradient - image2vector is commonly used in deep learning - np.reshape is widely used. In the future, you'll see that keeping your matrix/vector dimensions straight will go toward eliminating a lot of bugs. - numpy has efficient built-in functions - broadcasting is extremely useful


实验课件才是宝藏

主要的问题就是向量化的问题 要记清楚每个矩阵的行列

You might see that the training set accuracy goes up, but the test set accuracy goes down. This is called overfitting.

Choice of learning rate
```
If the learning rate is too large we may "overshoot" the optimal value. Similarly, if it is too small we will need too many iterations to converge to the best values. That's why it is crucial to use a well-tuned learning rate.
```

损失函数的推导还用到挺多概率论的内容，特别是极大似然估计

## Neural Network model and hidden layer

神经网络就是多了几层计算

得多动手写公式

sigmoid function只是其中的一个activation function
- tanh func 值域是[-1, 1]
- ReLu func (最推荐) 斜率大 学习速度快
- leaky ReLu func (最推荐)


为什么需要非线性激活函数：hiden unit用线性激活函数屁用没有,还不如不用

ReLu是分段函数 且其导数在0点是无定义的

cross entropy loss: 交叉熵损失

```py
cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
```

虽然说要不求甚解, 但还是不能太泛泛了，得聪明的学习，有用的东西用大功

Planar: 平面的

np.c_
```
np.c_ 是 NumPy 库中的一个函数，用于按列连接（column-wise concatenation）数组。它的作用是将两个或多个数组按列方向连接，生成一个新的数组。
```

sklearn provides simple and efficient tools for data mining and data analysis.
```py
# X:2 * 400 Y:1 * 400
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T[:, 0]);
'''
clf.fit(X, y)
X: 特征矩阵，形状为 (n_samples, n_features)，其中 n_samples 是样本数，n_features 是特征数。每一行代表一个样本，每一列代表一个特征。
y: 目标变量，形状为 (n_samples,)。它是一个一维数组，包含每个样本的目标值或类标签。

'''
```

np.meshgrid 
```
是 NumPy 中的一个函数，用于生成网格点坐标矩阵。通常在二维平面上创建坐标点网格时会用到这个函数。

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# 使用 meshgrid 生成坐标点网格
X, Y = np.meshgrid(x, y)
X 坐标轴上的坐标点：
[[1 2 3]
 [1 2 3]
 [1 2 3]]
Y 坐标轴上的坐标点：
[[4 4 4]
 [5 5 5]
 [6 6 6]]
```

np.arange
```
根据范围与步幅生成数组array
```

np.linspace
```
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
是 NumPy 中的一个函数，用于创建等间隔的一维数组。它的作用是生成指定区间内的固定数量的均匀分布的数值序列。
```

ravel 
```
函数是 NumPy 中的一个方法，用于将多维数组展平为一维数组。换句话说，它会将数组的所有元素按照其在内存中的顺序展开，生成一个新的一维数组。
```

```py
float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100)
算01的accuracy的方法 挺不错的
```

其实神经网络的构造方法和我们的学习很像：有一个初始参数，都是不断的调整，修正

列一般都是实验数量

np.multiply
```
执行元素级别的乘法操作的函数，该数组包含输入数组对应位置元素的乘积
```

np.dot
```
矩阵的乘法 向量的点乘
```

enumerate 
```
是 Python 中的一个内置函数，用于在迭代时同时返回索引和元素。 可以设置起始值
```

Tuning hidden layer size不一定能提高 还得找合适值 太大可能过拟合

**得学习一下它是怎么画图的啊**

有些内容还是值得看课的

随机初始化 否则会导致单元都在做一样的事 发挥不了神经网络的功效

## Deep Neural Network

超参数的选择还是非常经验性的

将NN中的unit与人类的神经元类比还是不太严谨，毕竟到现在都还没能知道人的一个神经元的作用是什么

其实NN就是找到一个函数可以正确的反映x到y的映射（监督学习）

# Improving Deep Neural Networks Hyperparameter tuning, Regularization and Optimization

Bias & Variance: 
- High Bias是欠拟合
  - 大一点的神经网络
  - 增加迭代次数
  - 更好的架构
- High Variance是过拟合
  - 增加数据
  - 正则化
  - 更好的架构

## Optimization
### 正则化

> L2 regularization: 就是平方和/总数*regularization param
regularization param

NN与logistics regression相比就是多了hidden layer, hidden units

不仅仅用在cost func 还要用在backpropagation中参数的更新故又称为weight decay

其实就是为了让学习的过程中调整Weight的值:
- 当regularization param很大时，为了降低cost func，W会逐渐变小，一些unit的影响就会变小，就会减低过拟合

但activation func是tanh时，当W小，z小时，activation func接近线性的表现，当每一层都是线性的时候，得到的映射也是线性的，所以不太可能过拟合

> dropout regular

在训练过程中，Dropout通过随机将神经元的输出置为零来随机丢弃一部分神经元，从而减少神经元之间的共适应性，提高模型的泛化能力。

dropout: during training, 随机设置一些activations to 0
  - 大约50%
  - 使network不要依赖于任何一个node

invert dropout: /keep-prob

在使用Inverted Dropout时，在每次训练迭代中，Dropout被应用于网络的隐藏层，其中每个神经元的输出值被随机地设置为零，以一定的概率保持不变。这个概率通常称为keep-prob，表示保留神经元的概率。

在测试阶段，为了保持模型的预测性能，不应该再进行随机丢弃神经元的操作。而是需要对训练阶段中Dropout造成的影响进行校正，以保持期望的输出值。为了实现这个校正，一种常见的做法是在训练阶段每个神经元的输出值乘以keep-prob。**这样可以保持训练和测试阶段的期望输出值一致。**

> other methods

data augmentation: 花式处理一下现有数据

early stopping: 缺点是没能用不同的方法解决两大问题

### Normalizing training sets

减去平均值 除以方差

同样的参数用在测试集

### exploding gradients and vanishing gradients

Exploding Gradients: 
- cause the weights of the neural network to update drastically, leading to instability in the training process.
- cause numerical overflow issues, making the training process practically infeasible.

Vanishing Gradients:
- This typically happens in deep networks where the gradients diminish as they are backpropagated through layers, especially in networks that utilize activation functions like sigmoid or tanh(因为导数都小于等于1).
- The problem with vanishing gradients is that they lead to very slow or no learning in the early layers of the network.(导数小更新的效率就低，或根本没更新) 
- It also makes it challenging for the network to learn long-term dependencies in sequential data.(既然更新不了，那权重就没啥变化，那么前面的层其实没啥卵用) 

> 解决方法

- Weight Initialization:
Proper initialization of weights can help alleviate both exploding and vanishing gradients. Techniques like Xavier/Glorot initialization or He initialization set initial weights such that they are conducive to stable training.

- Activation Functions:
Using activation functions like ReLU (Rectified Linear Unit) can help prevent vanishing gradients, as ReLU does not saturate in the positive region. Leaky ReLU, Parametric ReLU (PReLU), or Exponential Linear Unit (ELU) are variations that address the issue of dying ReLU neurons.
Avoiding activation functions like sigmoid and tanh can help prevent vanishing gradients, as they saturate in certain regions leading to gradient attenuation.

- Batch **Normalization**:(整个数据集一起处理)
Batch normalization helps to stabilize and speed up the training process by normalizing the inputs of each layer to have zero mean and unit variance. This can mitigate the issue of vanishing gradients by reducing internal covariate shift.

- Gradient Clipping:
Gradient clipping involves scaling the gradients when they exceed a certain threshold during training. This helps prevent the exploding gradient problem by limiting the size of the gradients.

- Residual Connections (ResNet):
Residual connections introduce skip connections that allow gradients to flow directly through the network, mitigating the vanishing gradient problem. Residual networks (ResNets) have shown great success in training very deep networks.

- Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU):
In the context of recurrent neural networks (RNNs), architectures like LSTM and GRU are designed to mitigate the vanishing gradient problem by introducing gating mechanisms that regulate the flow of information through the network.

- Gradient Regularization:
Techniques like L1 or L2 regularization can help prevent exploding gradients by penalizing large weights in the network, thereby encouraging the model to learn more robust features.

### Grad check

two-sided difference formula 来检测反向传播是否正确实现

because it takes into account information from both sides of the point at which the derivative is being approximated.

### Mini-batch gradient descent

迭代次数是一个循环 epoch
  the number of Mini-batch是一个循环
cost func在mini-batch中的gradient descent不是单调递减的，因为每次处理的都是不同的数据集

### other than gradient descent

> Exponentially weighted averages

序列化数据
就是同时通过权重保存前面的信息

将v_t视为前1/β天数据的平均值 β是权重  用到了指数的定义

v_t = (1-β)v_t-1 + βθ_t
v_0 = 0

不占内存 计算效率高

bias correction：因为v_0=0所以前期数据的偏差会比较大 
偏差修正：v_t / (1 - β^t)

> Momentum
Mini-batch算出的梯度可能会偏来偏去，较曲折, momentum利用exp weighted average来中和了一下

compute an exp weighted average of your gradient then use that gradient to update your weight instead(dw,db)

> RMSprop(root mean square prop)
和momentum很像 公式不同罢了

要确保不会出现除0情况 除个无穷小可能会爆炸

> Adam Optimization algorithm(Adaptive moment estimation)

即用到momentum也用到RMSprop 最后的update结合了两个算法的参数

### learning rate decay

有好多种方法
并不是首要选择

### Local optima

更容易碰到saddle point而不是local optima

plateaus can make learning slow

## Hyperparamter tuning

超参数：学习率1，layer num，hidden units num2，learning rate decay, mini-batch size2，momentum param2, Adam param...

try random value, coarse to fine

### appropriate scale

用线性标尺可能会导致随机的不随机

对数标尺：在对数坐标上取值：最小值的对数就是a，最大值的对数就是b

在一些比较敏感的区域增加取样点

### 超参数调整方式

Panda：一天天的跟踪调整one model
Caviar：Training many models in parallel

## Batch Normalization

不仅仅normalize X还normalize Z

就是将其均值变为0，方差变为1，其实也就是正态分布

当我们又不想都是正态分布（不然取值太局限了），所以在通过公式把Z转化一下，然后就引入了两个要学习的参数

用batch norm时B param没啥卵用

covariate shift

作用：
```
Imagine you're baking cookies. You have a recipe that tells you to add certain ingredients in specific amounts. Sometimes, though, the ingredients you have might vary in quality or quantity. Batch Normalization is like adjusting the ingredients to make sure each batch of cookies turns out consistently delicious.

In deep learning, each layer of a neural network receives inputs from the previous layer, just like ingredients in a recipe. During training, these inputs can fluctuate, making it hard for the network to learn effectively. Batch Normalization standardizes these inputs by adjusting and scaling them so that they have a consistent mean and variance.

By doing this, Batch Normalization helps the neural network learn more efficiently. It's like making sure the ingredients going into each layer of the network are of high quality and consistent, allowing the network to learn faster and produce better results.
```

测试的时候只有一个数据怎么用batch norm：将训练mini-batch时的均值方差通过指数加权平均来得到，然后放到test的时候用

## Softmax regression

generalization of logistics regression. 多分类 multi-class classification

输出的y_hat成了一个向量，向量的每一个位置代表了这个类别的概率

softmax activation function：确保向量的每一个元素非负且相加为1

没有hidden layer时决策边界很可能是线性的，当增加hidden layer时映射会变得复杂，就会成为非线性的

hard max: 最大为1 其他为0

Loss function：用到对数 The loss function of Softmax regression is the cross-entropy loss, which is commonly used in conjunction with the Softmax function for classification tasks

## 框架
open source with good governance

# Structuring Machine Learning Projects

## ML strategy

优化ideas:
- more data
- more diverse training set
- train longer
- try Adam instead of gradient descent
- try bigger/small network
- try dropout
- Add L2 regularization
- Network architecture
  - activation func
  - number of hidden units

### Orthogonalization(正交化)

正交化：一个方法只解决好一方面问题

### single number evaluation metric

一个指标可以告诉你你的操作是有利的还是无效有害的

**Precision**: Precision is about being precise or exact. In our cat example, precision tells us of all the pictures our model says have cats, how many actually have cats."When the model says there's a cat, how often is it right?"
**Recall**: Recall is about completeness. It tells us of all the pictures that actually have cats, how many did our model correctly identify."Out of all the pictures with cats, how many did the model find?"

两个指标不太好评价模型的优劣程度，F1 Score是两者的结合，一个指标好判断点，也方便我们迭代

### satisfaction and optimizing metrics

optimizing: 越高越好
satisfaction: 指达到某个门槛就好

### training, dev, test set

dev set是用来评估模型的
test set: evaluate your final cost bias

洗牌

### 什么时候改改变metrics

两个判断是不是猫的model，一个error rate低，但不能很好的判断色情图片，一个error rate高一点，但可以判断色情图片, 然而出现色情图片是不能忍受的

我们要改变metrics：增加识别错色情图片的cost权重

if doing well on ur metric + dev/test set doesn't correspond to doing well on your application, change ur metric and/or dev/test set

### compare to people

Bayes optimal error: 理论上的最佳值

avoid bias: 把human performance作为Bayes optimal error，其与training error的差值就是avoid bias，即可以提高的程度

"跳过了一些目前觉得没什么用的内容"

### transfer learning

**Transfer learning in deep learning is a method where a model trained for one task is reused as the starting point for a model on a second task.** This approach is popular in deep learning, especially in computer vision and natural language processing, as it allows for faster training and improved performance by leveraging pre-trained models.
**Transfer learning makes sense when there is a related task with abundant data available, or when pre-trained models are accessible, either speeding up model development or enhancing performance.** It is particularly useful when you have limited data for the second task, as it enables the model to learn more effectively by leveraging features learned from the first task.

### multi-task learning

例如一个图片识别model同时训练识别车辆，行人，红绿灯的能力，这时y_i就变成了一个向量，每一个位置上的损失都要计算到

some of the earlier features in NN can be shared between these different types of objects

也可以处理一些数据不完整的数据集：如一些样本，只标注了是否有人和车，没有标注有没有红绿灯

when make sense:
- **training on a set of tasks that could benefit from having shared lower-level features**
- amount of data you have for eash task is quite similar
- can train a big enough NN to do well on all the tasks.

### end-to-end learning

End-to-end learning in deep learning refers to a machine learning technique where a single neural network is trained for complex tasks using raw input data without the need for manual feature engineering or intermediate steps. 

**The main challenges** of end-to-end learning include the need for large amounts of training data and the lack of model interpretability, especially in critical domains like healthcare where data acquisition is challenging


face recognition(two steps):
- find where is the face?
- zoom in on the face and compare the face with the database

把一个大问题拆分为多个更容易解决的小问题

sometimes if you don't have large amount of data, hand-designed components may be useful.

# Convolutional Neural Networks

problem: 数据量太大

## edge detection

# MIT 6.S191

## Intro(binary classification)

AI(any tech that enable computers to mimic human behavior) 
> ML(Ability to learn without explicitly being programmed) 
> DL(**Extract patterns from data** using neural networks)

三个必要条件
- 大数据
- 硬件
- 软件

perceptron：forward propagation

common Activation Functions(居然还是这几个 得知道其导数和函数)
- sigmoid
- tanh
- ReLU

为什么要non-linear，因为大多数的数据是non-linear的，而权重乘数据得到的是linear的，要识别其pattern也需要non-linear

the empirical loss measures the total loss over out entire dataset

**cross entropy loss** can be used with models that output a probability between 0 and 1 (又是老朋友了)

**mean squared error loss** 均方误差损失 就是平方差 can be used with regression models that output continuous real numbers

优化：
- learning rates 多试或者设计一个adaptive版本的
    - adaptive learning rate 有好多现有的算法

- stochastic gradient descent: 取mini-batch做梯度下降然后取均值 smoother convergence allows for larger learning rates

underfitting: model does not have capacity to fully learn the data
overfitting: too complex, extra parameters does not generalize well

regularization(正则化): 解决overfit
- dropout: during training, 随机设置一些activations to 0
    - 大约50%
    - 使network不要依赖于任何一个node
- early stopping

## Deep Sequence Modeling(time)

many to one: sentiment classification
one to many: image captioning
many to many: machine translation

RNN: Recurrent neural networks

将input根据时间来分成不同的time step，并以此来生成不同时间的output: 不同时间的input可能有相关性，不能随意分

<img src="./img/rnn1.png">
<img src="./img/rnn2.png">
<img src="./img/rnn3.png">


可以用_+(矩阵维度)的标识，非常不错

设计标准 准则：
- 处理variable-length sequences
- track long-term dependencies
- 保存关于order的信息
- share parameters across the sequence


encoding language for a neural network

embedding(transform indexs into a vector of fixed size):
- one-hot embedding vector的size是词汇的数目 用index来标明vector 发现不了语义 不好
- learned embedding

backpropagation through time

<img src="./img/rnn4.png">

最后要修正的参数是h0 有两个问题：exploding gradients and vanishing gradients

vanishing gradients特别是针对long-term dependencies
- trick1：activation functions ReLu当x>0时，导数都是1，所以不会变小
- parameter init 初始化W为单位矩阵 biases为0
- Gated Cells：use gates to selectively add or remove information within each recurrent unit with gated cell 实例LSTMs(long short term memory)

<img src="./img/rnn5.png">

limitations:
- encoding bottleneck 信息可能会损失
- slow, no parallelization
- not long memory

> **Attention**

self-attention: 找到最重要的部分
- 查找
    - compute attention mask: how similar is each key to the desired query
- 提取
    - return the values highest attention

self-attention with nn
- data is fed in all at once! need to encode position info to understand order


<img src="./img/rnn6.png">
<img src="./img/rnn7.png">
<img src="./img/rnn8.png">

Softmax函数

<img src="./img/rnn9.png">

Google Colaboratory 是个做实验的好地方

## Lab 1: Intro to TensorFlow and Music Generation with RNNs

TensorFlow is a software library extensively used in machine learning. 

```py
sport = tf.constant("Tennis", tf.string) # 创建变量
print("`sport` is a {}-d Tensor".format(tf.rank(sport).numpy())) # 0-d  scalar

sports = tf.constant(["Tennis", "Basketball"], tf.string) # 1-d shape = 2
print("`sports` is a {}-d Tensor with shape: {}".format(tf.rank(sports).numpy(), tf.shape(sports)))

```
在TensorFlow中，tf.rank()返回张量的秩（即维度的数量），但它返回的是一个张量，而不是一个普通的整数值。.numpy()方法用于将张量转换为NumPy数组，这样就可以获取其具体的值。

in future labs involving image processing and computer vision, we will use 4-d Tensors. Here the dimensions correspond to the number of example images in our batch, image height, image width, and the number of color channels.

Use tf.zeros to initialize a 4-d Tensor of zeros with size 10 x 256 x 256 x 3.

You can think of this as 10 images where each image is RGB 256 x 256.

tf.zeros()
```
shape = (10, 256, 256, 3)
tf.zeros(
    shape,
    dtype=tf.float32,
    name=None
)
```

tf.constant()
tf.add()
tf.matmul() 矩阵乘法
tf.sigmoid()

TensorFlow uses a high-level API called **Keras** that provides a powerful, intuitive framework for building and training deep learning models.

Tensors can flow through abstract types called **Layers** -- the building blocks of neural networks. Layers implement common neural networks operations, and are used to update weights, compute losses, and define inter-layer connectivity. 
```py
### Defining a network Layer ###

# n_output_nodes: number of output nodes
# input_shape: shape of the input
# x: input to the layer

class OurDenseLayer(tf.keras.layers.Layer):
  def __init__(self, n_output_nodes):
    super(OurDenseLayer, self).__init__()
    self.n_output_nodes = n_output_nodes

  def build(self, input_shape):
    d = int(input_shape[-1])
    # Define and initialize parameters: a weight matrix W and bias b
    # Note that parameter initialization is random!
    self.W = self.add_weight("weight", shape=[d, self.n_output_nodes]) # note the dimensionality
    self.b = self.add_weight("bias", shape=[1, self.n_output_nodes]) # note the dimensionality

  def call(self, x):
    '''TODO: define the operation for z (hint: use tf.matmul)'''
    z = tf.add(tf.matmul(x,self.W), self.b)

    '''TODO: define the operation for out (hint: use tf.sigmoid)'''
    y = tf.sigmoid(z)
    return y

# Since layer parameters are initialized randomly, we will set a random seed for reproducibility
tf.keras.utils.set_random_seed(1)
layer = OurDenseLayer(3)
layer.build((1,2))
x_input = tf.constant([[1,2.]], shape=(1,2))
y = layer.call(x_input)

# test the output!
print(y.numpy())
mdl.lab1.test_custom_dense_layer_output(y)
```

**TensorFlow has defined a number of Layers that are commonly used in neural networks, for example a Dense**

Sequential模型是Keras中最简单的模型类型之一，它按顺序堆叠层

Model class, which groups layers together to enable model training and inference.
```py
### Defining a model using subclassing ###

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class SubclassModel(tf.keras.Model):

  # In __init__, we define the Model's layers
  def __init__(self, n_output_nodes):
    super(SubclassModel, self).__init__()
    '''TODO: Our model consists of a single Dense layer. Define this layer.'''
    self.dense_layer = tf.keras.layers.Dense(n_output_nodes)
  # In the call function, we define the Model's forward pass.
  def call(self, inputs):
    return self.dense_layer(inputs)
```

**Automatic differentiation** is one of the most important parts of TensorFlow and is the backbone of training with **backpropagation**. We will use the TensorFlow GradientTape ```tf.GradientTape``` to trace operations for computing gradients later.
```
### Gradient computation with GradientTape ###

# y = x^2
# Example: x = 3.0
x = tf.Variable(3.0)

# Initiate the gradient tape
with tf.GradientTape() as tape:
  # Define the function
  y = x * x
# Access the gradient -- derivative of y with respect to x
dy_dx = tape.gradient(y, x)

assert dy_dx.numpy() == 6.0
```
By default, the tape is discarded after it is played backwards;

是啊 损失函数的最小值点应该挺好找的

## Lab 1: Intro to TensorFlow and Music Generation with RNNs

ABC notation is a shorthand form of musical notation for computers.

comet.ml: Add two lines of code to your notebook or script and automatically start tracking code, hyperparameters, metrics, and more, so you can compare and reproduce training runs. 跟踪训练数据的

Breaking the problem down, what we're really asking the model is: given a character, or a sequence of characters, what is the most probable next character? We'll train the model to perform this task.

zip() 函数将两个可迭代对象打包成一个元组的迭代器
```py
names = ['Alice', 'Bob', 'Charlie']
ages = [30, 25, 35]

for name, age in zip(names, ages):
    print(name, 'is', age, 'years old')

```


repr(char) 将字符 char 转换为它的表示形式，并通过该格式化字符串将其填充到长度为 4 的字段中。

这里的映射就是很简单每个字符一个数字的embedding形式 即one-hot embedding vector

For each input, the corresponding target will contain the same length of text, except shifted one character to the right.

To do this, we'll break the text into chunks of seq_length+1. Suppose seq_length is 4 and our text is "Hello". Then, our input sequence is "Hell" and the target sequence is "ello".

np.random.choice() 
```py
# 函数是 NumPy 库中用于从给定的一维数组或整数生成随机样本的函数。它的作用是从指定的一维数组中随机抽取样本，或者从指定的整数范围中随机抽取整数，生成一个随机样本数组。
sample = np.random.choice(10, size=5)  # 从0到9中随机抽取5个整数
print(sample)
```

处理数据 建造模型

layers
- **tf.keras.layers.Embedding**: 是 TensorFlow 中的一个层，用于将整数编码的词汇表映射到密集的实数向量空间中
```
具体来说，Embedding 层的作用是将一个整数索引的词汇表映射到一个低维的实数向量空间中。这个映射是通过学习得到的，它会根据训练数据中的上下文关系来学习词汇表中每个词的表示。这样做的好处是，词汇表中的相似词在向量空间中也会有相似的表示，这使得模型可以更好地理解和泛化文本数据。

选取合适的 output_dim 参数值通常需要根据具体的任务需求和数据特征来进行调整。以下是一些常见的策略和建议：

- 任务需求： 根据你的任务类型和目标来选择 output_dim 的值。例如，对于一些简单的分类任务，可以选择较小的 output_dim 值；而对于复杂的语义理解任务，可能需要更大的 output_dim 值以获得更丰富的词汇表示。

- 词汇表大小： 通常情况下，output_dim 的值应该小于词汇表的大小。如果 output_dim 大于词汇表的大小，可能会导致过拟合或者损失信息。

- 数据特征： 考虑你的数据集中词汇的复杂度和多样性。如果你的数据集包含大量的词汇，并且这些词汇之间的关系复杂，可能需要选择较大的 output_dim 值来捕获更多的语义信息。

- 实验调优： 可以通过尝试不同的 output_dim 值并评估模型性能来进行实验调优。可以使用交叉验证或者验证集来评估不同 output_dim 值对模型性能的影响，并选择性能最好的值。

- 预训练模型： 如果你使用的是预训练的词向量模型（如Word2Vec、GloVe等），通常可以直接使用其提供的预训练嵌入向量，并选择与预训练模型相同的维度作为 output_dim 的值。
```
- tf.keras.layers.LSTM: Our LSTM network, with size units=rnn_units.
```
LSTM 层的主要作用是对输入序列进行处理，然后输出一个隐藏状态序列。这个隐藏状态序列可以继续传递给其他层进行进一步的处理，例如全连接层或输出层。
```
- tf.keras.layers.Dense: The output layer, with vocab_size outputs.

<img src="./img/rnn10.png">

model.summary(): 输出模型的一些信息
```
这里的32是batch size
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_1 (Embedding)     (32, None, 256)           21248     
                                                                 
 lstm (LSTM)                 (32, None, 1024)          5246976   
                                                                 
 dense (Dense)               (32, None, 83)            85075     
                                                                 
=================================================================
Total params: 5353299 (20.42 MB)
Trainable params: 5353299 (20.42 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

在 model.summary() 输出的模型结构中，"Param" 列表示每一层的参数数量。这些参数是神经网络中需要学习的权重参数和偏置参数的总数。

具体来说，对于不同类型的层，参数数量的计算方式如下：

全连接层（Dense）：参数数量等于输入特征维度乘以输出特征维度，再加上输出特征维度个偏置参数。例如，一个输入特征维度为 input_dim，输出特征维度为 units 的全连接层，参数数量为 (input_dim * units) + units。 其实就是w和b的维度 确实是要调整这么多 只不过我们是用矩阵来处理了

卷积层（Convolutional）：参数数量等于每个卷积核的大小（即权重参数）乘以卷积核的数量，再加上每个卷积核对应的偏置参数数量。例如，一个卷积核大小为 (kernel_size, kernel_size)，数量为 filters 的卷积层，参数数量为 (kernel_size * kernel_size * input_channels * filters) + filters。

循环层（Recurrent）：参数数量取决于循环神经网络（RNN）的类型。对于 LSTM 层，参数数量较复杂，但可以粗略地计算为 (4 * units * (units + input_dim + 1))，其中 4 * units 是每个 LSTM 单元的权重参数数量，units + input_dim + 1 是每个 LSTM 单元的偏置参数数量。

总的来说，"Param" 列显示了每一层的可学习参数的总数，这些参数需要在训练过程中通过反向传播算法进行学习。
```

预测和build这边不太懂啊

tf.random.categorical()
```py
# 定义骰子的概率分布
probabilities = [0.1, 0.2, 0.3, 0.15, 0.1, 0.15]

# 使用 tf.random.categorical 函数抽取一个样本 log以便对概率进行对数转换，这有助于提高数值稳定性
sample_index = tf.random.categorical(tf.math.log([probabilities]), num_samples=1)

# 由于 tf.random.categorical 返回的是一个 TensorFlow 张量，我们需要通过 .numpy() 方法将其转换为 Python 数值
sample_index = sample_index.numpy()[0]

# 打印抽取到的样本索引
print("抽取到的样本索引为:", sample_index)
```

我这里猜想因为pred的shape是(batch_size, sequence_length, vocab_size)
这里的vocab_size里是当前字符的下一个字符的可能性数组


```py
sampled_indices = tf.random.categorical(pred[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
sampled_indice
```
这个的目的是通过分类分布找到当前batch里的这一段sequence的预测值

RNN问题变成了一个标准分类问题

tf.keras.metrics.sparse_categorical_crossentropy 
```
是 TensorFlow 中用于计算稀疏分类交叉熵的函数。在深度学习中，交叉熵通常用作损失函数，用于衡量模型预测与实际标签之间的差异。稀疏分类交叉熵适用于标签以整数形式表示的情况，例如分类任务中的单标签分类。

from_logits 参数在 tf.keras.metrics.sparse_categorical_crossentropy 中用于指示输入的预测值是否是经过 softmax 函数处理的概率分布，或者是未经处理的模型输出（即 logits）。具体来说，它是一个布尔值，用于确定输入的 y_pred 参数是否表示原始的模型输出。

如果 from_logits=False（默认值），则函数假定 y_pred 是经过 softmax 处理后的概率分布，它会首先将其转换为 logits，然后计算交叉熵。这种情况下，y_pred 应该是一个概率分布，每个样本的所有类别概率之和应该为 1。

如果 from_logits=True，则函数假定 y_pred 是模型的原始输出，即未经 softmax 处理的 logits。在这种情况下，函数会直接使用这些 logits 来计算交叉熵，而不需要进行额外的转换。这对于提高计算效率或者避免数值稳定性问题（例如，softmax 计算可能会遇到数值上溢或下溢的情况）都很有用。
```

Cross-Entropy Loss就是吴恩达的分类问题里用的损失函数

Experiment are the core objects in Comet and will allow us to track training and model development. 

optimizer: Adam and Adagrad
```
tf.keras.optimizers.Adam 是 TensorFlow 中的一种优化器，它实现了 Adam 优化算法。Adam 是一种常用的自适应学习率优化算法，它结合了动量优化和自适应学习率的思想，在训练神经网络时表现良好。

Adam 优化算法的主要作用是调整模型的参数以最小化训练数据的损失函数。
```

超参数
```
params = dict(
  num_training_iterations = 3000,  # Increase this to train longer
  batch_size = 8,  # Experiment between 1 and 64
  seq_length = 100,  # Experiment between 50 and 500
  learning_rate = 5e-3,  # Experiment between 1e-5 and 1e-1
  embedding_dim = 256,
  rnn_units = 1024,  # Experiment between 1 and 2048
)
```


tqdm 
```
是 Python 中一个非常实用的库，用于在循环中显示进度条。它的名称来自于 "taqaddum"（阿拉伯语中的 "进展"），它为循环、迭代等操作提供了简单而有效的进度显示功能
```

由于 RNN 状态从一个时间步传递到另一个时间步的方式，模型在构建后只能接受固定的批量大小。

tf.expand_dims 
```py
# 是 TensorFlow 中的一个函数，用于在张量的特定轴上添加一个维度。其作用是在张量的指定位置增加一个新的维度，可以是在最外层也可以是在内部某个位置，从而改变张量的形状。

# 创建一个形状为 (3,) 的张量
x = tf.constant([1, 2, 3])

# 在第一个轴上添加一个维度
expanded_x = tf.expand_dims(x, axis=0)
# 现在 expanded_x 的形状为 (1, 3)
```

## Deep Computer Vision

"Regression vs Classification"

Neural style transfer: 根据照片生成不同风格的图片

都是先从人的角度开始考虑的

fully connected nn:
- input: one dimensional sequence
- 应用于CV的问题:
  - 失去了位置空间信息
  - 参数太多了

convolutional layer的优势：
- 参数少了
  - Param sharing
  - sparsity of connections: neuron connect patches of input. Only "sees" these values
> Convolution operation

using spatial structure

neuron connect patches of input. Only "sees" these values

用滑动窗口，每一个neuron只对于一个patch的input

<img src="./img/cnn1.png">

**The Convolution Operation**：image矩阵与filter矩阵 看两个矩阵的相似性就是element wise multiply 然后 sum up

矩阵乘法和Convolution Operation都挺神奇的

image * filter = feature map
<img src="./img/cnn2.png">

学习生成这些filter于是就推出了Convolutional Neural Networks
<img src="./img/cnn3.png">

filter的数量就是next的deepth
<img src="./img/cnn4.png">

> Padding

作用：
- Preservation of Spatial Information: ensuring that every pixel receives equal treatment during the convolution operation, which is crucial for maintaining spatial information and avoiding border effects
- Control of Output Size: 每次 Conv operation都会progressively reduce the spatial dimensions of feature maps
- Mitigation of Information Loss: 和第一点一样

类别：
- Same Padding：保证output feature maps and input images的维度一样
- Valid Padding：不做padding。This type is useful when reducing spatial dimensions is desired

filter matric 通常是维度是奇数的 
- 确保padding可以对称
- 有个中心点

> strided convolution

feature map的矩阵维度计算 我不想记 用到再说吧 非整时向下取整

<img src="./img/cnn_notions.png">

前面我们用的convolution operation实际上是cross-correlation
严格的convolution operation应该在最开始做一次镜像操作 即对/对角线进行镜像 为了结合律 (A*B)*C = A*(B*C)

> convolution over volumes
img: height:weight:channel
和2d没有实质上的区别，就是filter也变3d了，然后3d范畴的操作

multiple filters: 对input做多次convolution operate然后将feature map堆叠起来成x*x*num of filters的cube

第三维度：channel或depth

卷积网络其实是用来训练filter的以便可以更好的提取出特征
- 不管你图像多大 参数是确定的 就是filter的个数*维度 可以避免过拟合

一些符号标注都挺好理解的 没记

> non-linear operation ReLu apply after every convolution operation

ReLu为什么这么流行
```
ReLU（Rectified Linear Unit）函数在卷积神经网络（CNNs）中很流行，主要有以下几个原因：

mitigate the vanishing gradient problem

非线性： ReLU 是一个非线性函数，它能够引入非线性特征映射到神经网络中。这种非线性可以帮助网络学习更加复杂的函数关系，从而提高模型的表达能力。

稀疏激活性： 当输入为负数时，ReLU 的输出为零，这意味着它对负值进行了阈值化处理。这种稀疏激活性有助于减少梯度消失问题，因为它可以使部分神经元处于非活跃状态，从而缓解了梯度的衰减。

计算效率： ReLU 函数的计算简单高效，它只需要比较输入是否大于零并返回相应的值。与其他激活函数相比，如 Sigmoid 和 Tanh 函数，ReLU 函数的计算更加轻量级，有助于加速模型的训练和推断。导数是常数很简单好算 斜率大梯度下降也很快

抑制过拟合： 由于 ReLU 函数的稀疏激活性质，它能够在一定程度上对网络进行正则化，有助于防止过拟合的发生。

生物学解释性： ReLU 函数的形式更接近于神经元的生物学激活模式，因此在某种程度上更加符合生物学直觉。
```

> Pooling

max pool with 2*2 filter and stride 2
- reduced dimensionality
- spatial invariance

mean pooling, average pooling

pooling的超参数不需要学习

> 总结

通过一次又一次的使用来获取hierarchical decompositions of features

用文献中表现较好的超参数以及架构

<img src="./img/cnn5.png">

probability distribution properties:(不就概率分布吗 性质我还是记得的)
<img src="./img/cnn6.png">

这个架构类似于LeNet-5
```
LeNet-5 was introduced in 1998 by Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner

The architecture was initially designed for recognizing handwritten digits and document recognition, showcasing the power of gradient-based learning techniques
```
> LeNet-5


给我们的知识就是引入了类似上面的简单架构

精读section 2，泛读3
其他段讲了一些思路现在都还没应用起来(graph transformer network)

> Alexnet

https://www.analyticsvidhya.com/blog/2021/03/introduction-to-the-architecture-of-alexnet/

innovation:
- Depth of the Model:  eight layers, including five convolutional layers and three fully connected layers. similarity to LeNet-5, but much bigger. 
- Utilization of GPUs: use multiple GPUs
- 用了ReLu: mitigate the vanishing gradient problem and improved both training performance and computational efficiency compared to traditional activation functions like tanh and sigmoid
- Data Augmentation and Dropout
- Overlapping Max Pooling


new type of layer: local response normalization




> VGG-16

innovation
- Increased Depth and Layer Configuration:  utilizing 16 layers (hence the name) with 13 convolutional layers and 3 fully connected layers
- Use of Small Filter Sizes: consistent use of small 3x3 filter sizes throughout the network, replacing larger kernel sizes used in previous architectures like AlexNet. 
- Standardized Architecture: 每次conv都使depth翻倍，每次pooling都使H&W/2

Its pre-trained models have been widely used as a starting point for new projects, accelerating model development and deployment


> ResNet: residual network

deep的NN不好训练: gradient explode or gradient vanish

skip connections: learning functions as F(x) + x instead of just F(x). 

residual block: which consist of convolutional layers followed by batch normalization and ReLU activation functions. 
plain network + skip connection = ResNet

矩阵维度不同时再加一个Weight Matrix

> Inception network/layer

让网络学习filter的size以及combination

可以使用1*1 conv来降低depth，然后再复原，这样可以减少计算开销，而通过1*1conv生成的layer也被称为`bottleneck layer`

don't want to have to decide what size of pooling layers to use, inception module, says, let's do them all and concatenate the results.

<img src="./img/inception1.png">

<img src="./img/inception2.png">

inception现在发展了很多版本

学习整合模块的方法：大量读文献

可以去github上看看一些NN的开源实现，并以此为基础

> 1 * 1 conV

对于3d conv比较有用 其实有点像fully connected 增加了一个非线性函数 然后保持数据维度不变
也叫network in network

可以用来降低计算维度

> transfer learning

imageNet

特征向量

根据自己数据集的大小freeze一些layer，train一些layer

> data augmentation

common augmentation method:
- mirroring
- random cropping
- rotation
- shearing
- local warping 

color shifting: "PCA color augmentation"

implementing distortions during training: 用线程

也有一些超参数

> 现状

two sources of knowledge
- labeled data(supervise learning)
- hand engineered features/network architecture/other components

tips for doing well on benchmarks
- ensembling: 训练多个网络对结果取平均值
- multi-crop at test time: run classifier on multiple versions of test images and average results 

> Object Detection

solution
- Naive solution: 将图片切割一个又一个的box，然后分类，要是有类别就输出打印
- R-CNN algorithm: find regions that we think have objects. Use CNN to classify. 就是找颜色复杂 信息多的区域 人为选择区域? 但还是太慢了 R代表Region
- fast R-CNN
<img src="./img/cnn7.png">

> Segmentation


不仅仅是框出来了，要直接知道边界



<img src="./img/cnn8.png">
<img src="./img/cnn9.png">

## Deep Generative Modeling

不是大语言模型

supervised: learn function to map data to labels
- classification
- regression
- object detection

unsupervised learning: learn the hidden or underlying structure of the data
- clustering
- feature or dimensionality reduction

goal of generative modeling: 主要目标是学习数据的分布，然后可以利用这个学习到的分布来生成新的数据样本。

举个例子，假设我们有一个包含很多人脸照片的数据集，我们可以使用生成模型来学习这些人脸照片的分布，然后可以利用学到的分布来生成新的人脸照片，这些照片看起来与原始数据集中的照片非常相似，但实际上是全新生成的。

debiasing 提高模型的泛化能力
```
生成模型的去偏能力指的是生成模型在生成数据时减少或消除数据中的偏见（bias）的能力。这种偏见可能是由于数据集本身的不平衡、样本选择偏差或者数据采集过程中的偏差等原因导致的。

以下是几种生成模型的去偏方法：

数据预处理：在训练生成模型之前，可以对数据进行预处理，如样本重加权、过采样或欠采样等方法，以平衡数据集中不同类别之间的样本分布。

使用公平性约束：在训练生成模型时，可以引入公平性约束，确保生成的数据不会体现出特定的偏见。这可以通过在模型的损失函数中引入公平性相关的项来实现，例如差异对待的约束或平衡性约束。

多模态生成：生成多个与不同特征相关的模态（mode）可以帮助减少数据偏见。例如，在生成图像时，可以生成多个不同肤色、性别或年龄的样本。

生成对抗网络（GAN）的应用：生成对抗网络是一种强大的生成模型，可以通过对抗训练的方式学习生成真实样本分布的模型。在生成过程中，可以引入公平性的约束，使生成器不会产生偏见的样本。

后处理技术：在生成数据之后，可以使用后处理技术来进一步减少偏见。例如，可以通过重新加权或者修正生成的样本来消除偏见。
```

outlier detection(处理特殊情况)

> Latent variable models

Latent variable
```
指在模型中未直接观察到的但可以通过观察到的变量来推断或间接推断的变量。

想象一下你在照一张照片，然后上传到社交媒体上。你能看到照片中的颜色、形状、纹理等各种信息，但是你无法直接观察到照片中的拍摄时间、地点、摄影师的情绪等。这些你无法直接观察到的信息就可以被看作是潜在变量。

在潜在变量模型中，我们尝试通过已知的数据来推断或估计这些未观察到的潜在变量。例如，在照片的例子中，我们可以通过照片的像素值和颜色分布等信息，来推断拍摄时间或地点。

潜在变量模型在许多领域都有应用，例如在机器学习中，用于无监督学习任务，如聚类、降维等；在社会科学中，用于隐性变量的研究，如心理学中的隐性特质；在自然语言处理中，用于主题模型等任务。
```

Autoencoders: learns mapping from the data x to a low-dimensional latent space z
```
想象一下你有一个神奇的工具箱，可以帮你把东西塞进去，然后把它们重新取出来。但这个工具箱会变得更聪明，它会学习如何更有效地把东西塞进去和取出来，以便在取出来的时候你可以得到比之前更有用的东西。这个神奇的工具箱就好像是Autoencoders。

Autoencoders是一种神经网络架构，它们被用来学习如何有效地表示数据，即将输入数据编码成一个低维度的表示，然后再将这个表示解码回原始数据。它们由一个编码器和一个解码器组成，编码器将输入数据映射到潜在空间中，而解码器则将潜在空间中的表示映射回原始数据空间。

这种编码和解码的过程使得Autoencoders可以学习数据中的特征和模式，并且可以生成类似原始数据的新样本。它们通常被用来进行数据压缩、特征提取、数据去噪、生成新数据等任务。

例如，在图像处理中，你可以使用Autoencoders来压缩图像数据并提取关键特征，也可以用它们来去除图像中的噪声或生成新的图像。在自然语言处理中，你可以使用Autoencoders来学习文本数据的低维度表示，以便进行文本分类、情感分析等任务。
```

<img src="./img/gm1.png">

deterministic

encoder: 包含一些hidden layer可以是fully connected，或convolutional
bottleneck: 就是低维数据 latent space representation. dimension 不同时有着不同的效果 dimension越多越精确
decoder: 

data specific, learned compression

denoising autoencoder: 为了将noisy input reconstruct为denoised image

neural impeding

variational autoencoders(VAEs)  bottleneck不再是vector了而是distribution variational指会生成new images similar to the data but not forced to be strict reconstructions
<img src="./img/gm2.png">
<!-- <img src="./img/gm4.png"> -->
<img src="./img/gm5.png">

trying to enforce that each of those latent variables adapts a probabilities distribution that's similar to that prior. 一般是正则高斯分布

<img src="./img/gm6.png">
<img src="./img/gm7.png">

<img src="./img/gm3.png">

```
当我们训练Variational Autoencoders（VAEs）时，我们希望模型学会将输入数据映射到一个潜在空间中，并且这个潜在空间的分布要符合某种我们预先设定的标准，比如高斯分布。KL散度损失用来度量模型学习到的潜在空间的分布与我们期望的标准分布之间的差异。

让我们用一个简单的例子来说明。假设你正在训练一个VAE来学习人脸图像的潜在表示。你希望这个潜在表示的分布是一个标准正态分布，即均值为0，方差为1的高斯分布。但是在训练过程中，模型可能学会了一个潜在表示的分布，它的均值和方差可能与你期望的标准分布有一些差异。

KL散度损失就是用来衡量这种差异的。它会计算模型学习到的潜在表示分布和期望的标准分布之间的差异，然后将这个差异作为损失的一部分加到模型的总损失中。通过最小化KL散度损失，模型会更好地学会将潜在表示分布调整为符合我们期望的标准分布。

总的来说，KL散度损失就是帮助模型学会将学到的潜在表示分布调整为我们期望的标准分布，从而提高模型的性能和可解释性。
```
problem: 通过mean vector以及standard deviation vector生成sampled latent vector，不能对sampled latent vector使用back propagation

reparameterization trick:
<img src="./img/gm8.png">

<img src="./img/vae2.png">
<img src="./img/vae3.png">

<img src="./img/gm9.png">


disentangled VAEs: different neurons in latent distribution are uncorrelated and all try learn something different. 当loss function加上β参数后

<img src="./img/gm10.png">

公式有点不一致啊 得确认一下: 课上的公式才是正确的

<img src="./img/gm11.png">

### GANs(Generative Adversarial Networks)

<img src="./img/gans1.png">

那其实直接最终目的是生成和real data一模一样的数据，但其实我们只是学习其中的权重关系罢了，最终我们可以制造不同的input输入

<img src="./img/gans2.png">
nn.BCELoss 代表输入的input已经做过sigmoid
<img src="./img/gans3.png">

超分辨率重构

Discriminator network的loss有两部分，分别是与fake的比以及与true的比
> Advance 

progressive growing of GANs: 增加layer num

conditional GANs and pix2pix: paired translation

CycleGAN: domain transformation unpaired data

**Diffusion Models** 近年来最火的model, 更有想象力了，我们与机器的不同其实就是想象力
