import numpy as np
from sklearn.cluster import KMeans


# 1. 生成码书
def train_codebooks(features, num_codebooks=8, num_centroids=32):
    """
    训练码书
    :param features: 训练数据，形状为 (num_samples, 192)
    :param num_codebooks: 子向量数量（默认为8）
    :param num_centroids: 每个码书中的聚类中心数量（默认为32）
    :return: 一个包含所有码书的列表
    """
    subvector_size = features.shape[1] // num_codebooks
    codebooks = []

    for i in range(num_codebooks):
        sub_features = features[:, i * subvector_size: (i + 1) * subvector_size]
        kmeans = KMeans(n_clusters=num_centroids, random_state=0).fit(sub_features)
        codebooks.append(kmeans.cluster_centers_)

    return codebooks


# 2. 编码过程
def encode(features, codebooks):
    """
    使用码书对特征进行编码
    :param features: 输入特征，形状为 (num_samples, 192)
    :param codebooks: 码书列表
    :return: 编码后的整数索引，形状为 (num_samples, 8)
    """
    num_codebooks = len(codebooks)
    subvector_size = features.shape[1] // num_codebooks
    encoded_indices = np.zeros((features.shape[0], num_codebooks), dtype=np.int32)

    for i in range(num_codebooks):
        sub_features = features[:, i * subvector_size: (i + 1) * subvector_size]
        distances = np.linalg.norm(sub_features[:, np.newaxis, :] - codebooks[i], axis=2)
        encoded_indices[:, i] = np.argmin(distances, axis=1)

    return encoded_indices


# 3. 解码过程
def decode(encoded_indices, codebooks):
    """
    使用码书对编码的索引解码为原始特征的近似值
    :param encoded_indices: 编码后的整数索引，形状为 (num_samples, 8)
    :param codebooks: 码书列表
    :return: 解码后的特征，形状为 (num_samples, 192)
    """
    num_codebooks = len(codebooks)
    subvector_size = codebooks[0].shape[1]
    decoded_features = np.zeros((encoded_indices.shape[0], num_codebooks * subvector_size))

    for i in range(num_codebooks):
        decoded_features[:, i * subvector_size: (i + 1) * subvector_size] = codebooks[i][encoded_indices[:, i]]

    return decoded_features
