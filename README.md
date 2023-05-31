## 一、数据处理
&nbsp;&nbsp;&nbsp;&nbsp;CIFAR-100数据集包含60000张图片（32×32的像素），共20个大类，每一个大类有5个子类，每一个子类为600张图片。CIFAR-100已按比例划分训练集和测试集，其中有50000张训练集，10000张测试集。<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;1.首先对训练集进行预处理：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;进行填充，padding参数取4<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;以0.5的概率进行水平翻转<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;标准化Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))，使数据分布为[-1,1]<br>
&nbsp;&nbsp;&nbsp;&nbsp;2.再对测试集进行预处理：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;标准化Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))，使数据分布为[-1,1]<br>

## 二、模型设定
（一）网络结构<br>
&nbsp;&nbsp;&nbsp;&nbsp;针对CIFAR-100数据集，这里选用ResNet-18网络结构尝试图像分类。<br>

（二）数据增强<br>
&nbsp;&nbsp;&nbsp;&nbsp;拟分别采用mixup、cutmix、cutout三种数据增强方式对训练集进行尝试，再与未做数据增强的原始模型进行对比。<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;1.mixup<br>
&nbsp;&nbsp;&nbsp;&nbsp;从每个batch中随机选择两张图像，并以一定比例混合生成新的图像，采用混合后新图像进行训练，原始图像不参与训练过程，其中混合的比例从beta分布中随机选取。<br>
&nbsp;&nbsp;&nbsp;&nbsp;2.cutmix<br>
&nbsp;&nbsp;&nbsp;&nbsp;将图片的一部分区域裁剪掉，随机填充训练集中的其他数据的区域像素值，分类结果按一定的比例分配。而对于本数据集，采用0.5的概率随机使用cutmix方法进行尝试。<br>
&nbsp;&nbsp;&nbsp;&nbsp;3.cutout<br>
&nbsp;&nbsp;&nbsp;&nbsp;采用的操作是随机裁剪图像中的一块正方形区域，并以0值进行填充。剪掉的图像块数目默认设置为1，裁剪的正方形块边长设置为16。<br>

（三）批量处理<br>
&nbsp;&nbsp;&nbsp;&nbsp;对于每种数据增强方式的epoch，都设置为100。训练集的batch_size设置为128，一个epoch约有391个iteration；测试集的batch_size设置为100，一个epoch约有393个iteration。


（四）优化器<br>
&nbsp;&nbsp;&nbsp;&nbsp;优化器选用Adam优化，它是一种利用动量和缩放的自适应学习率优化算法，适用于非平稳目标和具有非常嘈杂或稀疏梯度的问题。在此处选用学习率0.01。<br>

（五）损失函数<br>
&nbsp;&nbsp;&nbsp;&nbsp;损失函数选用交叉熵损失，适合于分类问题。<br>

（六）评价指标<br>
&nbsp;&nbsp;&nbsp;&nbsp;考察分类的准确率，为分类正确的个数与样本个数的比值。<br>
