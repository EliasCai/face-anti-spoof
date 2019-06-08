# 人脸翻拍检测

1. feature_apply: 使用模型进行预测
2. feature_svm: 特征提取并生成数据保存
3. feature_train: 训练模型


# fisher.py
1. generate_gmm: 训练GMM模型，并保存N个kernel的均值、方差和权重
1.1 folder_descriptors: 遍历文件夹下的图片，并生成图像特征
1.1.1 image_descriptors: 对输入的图像生成SIFT特征
1.2 dictionary: 训练GMM模型，并返回均值、方差和权重
2. fisher_features: 遍历文件夹，按照文件夹返回fisher向量
2.1 get_fisher_vectors_from_folder: 遍历文件夹下的图片，生成fisher向量
2.1.1 image_descriptors: 对输入的图像生成SIFT特征
2.1.2 fisher_vector: 对输入的图像特征生成fisher向量
2.1.2.1 likelihood_statistics: 极大似然分析
2.1.2.1.1 likelihood_moment
2.1.2.2 fisher_vector_weights
2.1.2.3 fisher_vector_means
2.1.2.4 fisher_vector_sigma
2.1.2.5 normalize: 正态化特征
3. train: 训练分类模型