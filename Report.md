# 基于有监督机器学习的声源定位
## 一 、基于机器学习的定位方法
### 输入数据预处理：
>输入采样协方差矩阵

- 声压：<img src="./figure/equation/sound_pressure.png" width = "180"  alt="sound_pressure" align=center />

	其中，**_n_**是噪声，**_S(f)_**是声源信号， 
    **_g(f,r)_**是格林函数，接收阵有**_L_**个传感器

- 归一化：<img src="./figure/equation/norm_pressure.png" width = "130"  alt="norm_pressure" align=center />
    
    复声压归一化，取**Ns**个快拍的均值得采样协方差矩阵

- 采样协方差矩阵：<img src="./figure/equation/scm.png" width = "180"  alt="scm" align=center />

	最后，取采样协方差矩阵上三角每个元素的实部和虚部，组成 **_D=L*(L+1)_**维的列向量作为前馈神经网络的输入。

###声源位置映射：
>声源定位当作分类问题

将要搜索的Range划分成K个网格，一个声源水平位置对应一个标记向量**t<sub> n</sub>**，每一个向量只有一个元素是1，对应声源位置

<div  align="center">    
<img src="./figure/equation/map_label.png" width = "180"  alt="map_label" align=center />
</div>

其中，n=1,...,N是输入样本的序号,r<sub>k</sub>(k=1,...,K)是预先标记的水平位置，因此每一个输入样本的标记向量tn都对应一个水平位置rk。

<div  align="center">    
<img src="./figure/equation/map_matrix.png" width = "400"  alt="map_matrix" align=center />
</div>

<p align="center"> 图1 声源定位当作分类问题 </p>

###前馈神经网络:
>网络由输入、输出和一个隐含层构成，分别包含D，M，K个神经元

>神经网络的输出对应了声源水平位置的概率估计

<div  align="center"> 
<img src="./figure/equation/fnn.png" width = "400"  alt="fnn" align=center />
<img src="./figure/equation/z=f(vx)&y=(wz).png" width = "150"  alt="z=f(vx)&y=(wz)" align=center />
</div>  
<p align="center"> 图 2 前馈神经网络 </p>

_**W**_ , **_V_** 是待学习的神经网络系数，**_f_**, **_h_**为非线性函数，
<div  align="center"> 
<img src="./figure/equation/sigmoid_function.png" width = "400"  alt="sigmoid_function" align=center />
</div>  
<p align="center"> 图 3 **_Sigmoid&Softmax_**函数 </p>

神经网络训练的目标函数为：
<div  align="center"> 
<img src="./figure/equation/cost_function.png" width = "300"  alt="cost_function" align=center />
</div>  

##二、仿真和实验结果
>Matlab处理数据+Tensorflow训练神经网络

仿真环境为[SWell96Ex](http://swellex96.ucsd.edu/environment.htm),如下图：

<div  align="center">    
<img src="./figure/enviroment/environment.png" width = "500"  alt="environment" align=center />
</div>

<p align="center"> 图4 仿真环境示意图 </p>

###仿真设置
用来训练神经网络的声学数据由Kraken生成,仿真中Ns=10，L=21，D=441，M=441，K=300。训练集是1.82-8.65km之间的均匀采样的3000个数据样本，测试集为另外生成的300个数据样本，仿真中噪声设置为高斯白噪声。
####仿真数据训练和测试的结果
单频频率为109Hz，多频频率为109,232,385Hz
<div align="center">   
    <img src="./figure/result/simulation/simulation_rang_prediction@109Hz.png" width = "400"  alt="simulation_range_prediction@109Hz">
    <img src="./figure/result/simulation/multi_fre_simulation_rang_prediction.png" width = "400"  alt="multi_fre_simulation_rang_prediction">
</div>
<p align="center"> 图5 测试集声源Range预测(左：单频，右：多频)  </p>

<div  align="center">  
    <img src="./figure/result/simulation/accuracy_of_trained_model.png" width = "400" alt="accuracy_of_trained_model" >
    <img src="./figure/result/simulation/multi_fre_accuracy_of_trained_model.png" width = "400"  alt="multi_fre_accuracy_of_trained_model" >
</div>
<p align="center"> 图6 模型准确度与信噪比，训练步数的变化关系
(左：单频，右：多频)  </p>

<div  align="center">  
    <img src="./figure/result/simulation/simulation_learning_curve_4.png" width = "400"  alt="simulation_learning_curve_4" >
    <img src="./figure/result/simulation/multi_fre_simulation_learning_curve_4.png" width = "400"  alt="multi_fre_simulation_learning_curve_4" >
</div>
<p align="center"> 图7 学习曲线：交叉熵随训练步数的变化(左：单频，右：多频)  </p>

###实验数据训练和测试的结果
实验数据取自[SWell96Ex Event S5](http://swellex96.ucsd.edu/s5.htm)，数据处理的时候，取1s为一个快拍（无重叠）,每2个快拍求一次采样协方差矩阵，也即取Ns=2，其余参数同仿真数据训练部分。同样，图8~9中，单频频率为109Hz，多频频率为109,232,385Hz。

<div  align="center">  
    <img src="./figure/result/experimental/FNN on SWellS5 @ 109Hz.png" width = "400"  alt="FNN on SWellS5 @ 109Hz" >
    <img src="./figure/result/experimental/FNN on SWellS5 @109,232,385Hz.png" width = "400"  alt="FNN on SWellS5 @109,232,385Hz" >
</div>
<p align="center">图8 测试集声源Range预测(左：单频，右：多频)  </p>

<div  align="center">  
    <img src="./figure/result/experimental/learning curve on Event S5@109Hz.png" width = "400"  alt="learning curve on Event S5@109Hz" >
    <img src="./figure/result/experimental/learning curve on Event S5(@109,232,385Hz).png" width = "400"  alt="learning curve on Event S5(@109,232,385Hz)" >
</div>
<p align="center">图9 学习曲线：交叉熵随训练步数的变化(左：单频，右：多频)  </p>

<div  align="center">  
    <img src="./figure/result/experimental/accuracy of trained model compare.png" width = "400"  alt="accuracy of trained model compare" >
</div>
<p align="center">图10 模型准确度随训练步数的变化 </p>

##FNN定位的性能以及与传统匹配场处理方法的比较
<div  align="center">  
    <img src="./figure/result/experimental/FNN on SWellS5 @ 109Hz.png" width = "400"  alt="FNN on SWellS5 @ 109Hz" >
    <img src="./figure/result/experimental/MCE on SWellEx95S5 @ 109Hz(measure data as replica_line_pic_1.png" width = "400"  alt="MCE on SWellEx95S5 @ 109Hz(measure data as replica_line_pic" >
    <img src="./figure/result/experimental/Bartlett MFP on SWellEx95S5(measure data as replica)_line_pic.png" width = "400"  alt="Bartlett MFP on SWellEx95S5(measure data as replica)_line_pic" >
    <img src="./figure/result/experimental/Bartlett MFP on SWellS5 @109Hz.png" width = "400"  alt="Bartlett MFP on SWellS5 @109Hz" >
</div>
<p align="center">图11 FNN定位的性能以及与传统匹配场处理方法的比较（单频@109Hz） </p>
Note:

<div  align="center">  
    <img src="./figure/result/experimental/FNN on SWellS5 @109,232,385Hz.png" width = "400"  alt="FNN on SWellS5 @109,232,385Hz" >
    <img src="./figure/result/experimental/MCE on SWellEx95S5 @109,232,385Hz(measured replica)_line_pic.png" width = "400"  alt="MCE on SWellEx95S5 @109,232,385Hz(measured replica)_line_pic" >
    <img src="./figure/result/experimental/Bartlett MFP on SWellEx95S5 @109,232,385Hz(measured replica)_line_pic.png" width = "400"  alt="Bartlett MFP on SWellEx95S5 @109,232,385Hz(measured replica)_line_pic" >
    <img src="./figure/result/experimental/Bartlett MFP on SWellS5 @ 109,232,385Hz.png" width = "400"  alt="Bartlett MFP on SWellS5 @ 109,232,385Hz" >
</div>
<p align="center">图12
FNN定位的性能以及与传统匹配场处理方法的比较（多频点@109，232，385Hz） </p>
Note:

###声速剖面失配对定位结果的影响
图13，15，16的仿真中，快拍数Ns=10,信噪比SNR=5dB。
<div  align="center">  
    <img src="./figure/result/experimental/Three-frequency localization on SWellS5(SNR5L10,optimized as training,optimized,i905,i906 as test)FNN.png" width = "400"  alt="Three-frequency localization on SWellS5(SNR5L10,optimized as training,optimized,i905,i906 as test)FNN" >
    <img src="./figure/result/experimental/ssp3.png" width = "400"  alt="ssp3" >
</div>
<p align="center">图13 SSP失配对FNN定位结果的影响</p>
图中，Ns=10,SNR=5dB；(a),(d),(g):109,232,385Hz; (b),(e),(h):127,163,280Hz;(c),(f),(I):145,198,335Hz；(a),(b),(c):optimized;(d),(e),(f):i905;(g),(h),(i):i906。

1000次蒙特卡洛仿真绘制性能曲线：
<div  align="center">  
    <img src="./figure/result/experimental/Accuracy to SNR , FNN vs Bartlett & MCE.png" width = "400"  alt="Accuracy to SNR , FNN vs Bartlett & MCE" >
    <img src="./figure/result/experimental/Error to SNR , FNN vs Bartlett & MCE.png" width = "400"  alt="MCE on SWellEx95S5 @109,232,385Hz(measured replica)_line_pic" >
    <img src="./figure/result/experimental/Accuracy to SNR , FNN vs Bartlett & MCE_i906.png" width = "400"  alt="Accuracy to SNR , FNN vs Bartlett & MCE_i906" >
    <img src="./figure/result/experimental/Error to SNR , FNN vs Bartlett & MCE_i906.png" width = "400"  alt="Error to SNR , FNN vs Bartlett & MCE_i906" >
</div>
<p align="center">图14
FNN定位的性能曲线（多频点@109，232，385Hz） </p>
###混合不同ssp下采集数据进行训练的结果
<div  align="center">  
    <img src="./figure/result/experimental/Three-frequency localization on SWellS5(SNR5L10,optimized as training,optimized,i905,i906 as test)FNN.png" width = "400"  alt="Three-frequency localization on SWellS5(SNR5L10,optimized as training,optimized,i905,i906 as test)FNN" >
    <img src="./figure/result/experimental/ssp4.png" width = "400"  alt="ssp4" >
</div>
<p align="center">图15 混合不同ssp(optimized+i906)下仿真数据进行训练的结果</p>
图中，Ns=10,SNR=5dB；(a),(d),(g) :109,232,385Hz; (b),(e),(h):127,163,280Hz;(c),(f),(I):145,198,335Hz;(a),(b),(c) :optimized+; (d),(e),(f):i905;(g),(h),(I):i906i*。

<div  align="center">  
    <img src="./figure/result/experimental/combinevssingle_lef.png" width = "400"  alt="combinevssingle_lef" >
    <img src="./figure/result/experimental/combinevssingle_right.png" width = "400"  alt="combinevssingle_right" >
</div>
<p align="center">图16 混合数据训练和单一数据训练比较</p>
图中，Ns=10,SNR=5dB；Up to down:i905,i906*。

1000次蒙特卡洛仿真绘制性能曲线：
<div  align="center">  
    <img src="./figure/result/experimental/Accuracy to SNR , Combined vs Single.png" width = "400"  alt="Accuracy to SNR , Combined vs Single" >
    <img src="./figure/result/experimental/Error to SNR , Combined vs Single.png" width = "400"  alt="Error to SNR , Combined vs Single" >
</div>
<p align="center">图17
FNN定位的性能曲线（多频点@109，232，385Hz） </p>

##三、小结
