# relation-extraction
对论文Neural Relation Extraction with Selective Attention over Instances的改进：  
1、不再使用注意力加权，而是采用中间网络学习每个包内样本的共同特征；  
2、改进损失函数，增加对共同特征的学习能力   
3、基于3θ准则的高斯去躁   
4、在PCNN中引入LSTM，这里有很多其他的点可以做
