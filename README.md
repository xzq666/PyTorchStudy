# PyTorchStudy

## Pytorch核心组件
1、张量<br>
torch.Tensor是一个多维矩阵，其中包含单个数据类型的元素。它是框架的中央数据结构。可以从NumPy数组或列表创建Tensor，并执行各种操作，例如索引，数学，线性代数。
<br>
张量支持一些其他增强功能，从而使其具有独特性。除CPU外，它们还可以加载到GPU中进行更快的计算。并且它们支持形成一个向后图，该图跟踪使用动态计算图（DCG）应用于它们的每个操作以计算梯度。
<br>

2、Tensor的Autograd功能<br>
对于复杂的神经网络很难做微积分，在高维空间更是令人头脑混乱。幸运的是PyTorch中的Tensor具有Autograd，提供自动求导方法。
<br>
Tensor对象支持神奇的Autograd功能，即自动区分，这是通过跟踪和存储在Tensor流经网络时执行的所有操作来实现的。
<br>

3、nn.Module类，用于自定义神经网络<br>
在PyTorch中，通过将其定义为自定义类来构建神经网络，需要继承nn.Module类。这为神经网络类注入了有用的属性和强大的方法。

4、损失函数<br>
损失函数定义了神经网络的预测与真实情况之间的距离，而损失的定量度量则帮助驱动网络更接近对给定数据集进行最佳预测。PyTorch提供了用于分类和回归任务的所有常见损失函数。

5、优化器<br>
对权重进行优化以实现最低的损失是用于训练神经网络的反向传播算法的核心。PyTorch提供了大量的优化器来完成这项工作，这些优化器通过torch.optim模块公开。

<br>
使用上述组件，可以通过5个步骤构建神经网络：<br>
1、将神经网络构造为自定义类（从该类继承nn.Module），其中包含隐藏层张量以及forward通过各种层和激活函数传播输入张量的方法。<br>
2、使用此forward方法通过网络传播特征（从数据集）张量得到一个output张量。<br>
3、计算loss并使用内置的损失函数。<br>
4、传播的梯度loss使用自动分化能力（Autograd）与反向传播backward方法。<br>
5、使用损耗的梯度来更新网络的权重（这是通过执行优化器的一个步骤来实现的：optimizer.step()）。<br>
这五个步骤构成了一个完整的训练周期。
