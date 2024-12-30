import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
import numpy as np


class Location:
    def __init__(self,  xys:np.ndarray):
        
        self.model = KMeans(n_clusters=5)

        labels = self.model.fit_predict(xys)

        # 获取核心样本的索引
        core_samples_mask = np.zeros_like(self.model.labels_, dtype=bool)
        core_samples_mask[self.model.core_sample_indices_] = True

        # 获取每个聚类的核心样本
        core_samples = xys[core_samples_mask]
        core_labels = labels[core_samples_mask]

        # 计算每个聚类的中心值
        cluster_centers = []
        for cluster_id in np.unique(core_labels):
            if cluster_id!= -1:
                cluster_center = np.mean(core_samples[core_labels == cluster_id], axis=0)
                cluster_centers.append((cluster_id, cluster_center))

        # 根据中心值的某个维度（这里假设第一个维度）从大到小排序
        cluster_centers.sort(key=lambda x: x[1][0], reverse=True)

        # 重新分配clusterid
        new_cluster_ids = {old_id: new_id for new_id, (old_id, _) in enumerate(cluster_centers)}
        new_labels = np.array([new_cluster_ids[label] if label!= -1 else -1 for label in labels])

        # 将结果整理成DataFrame
        data = pd.DataFrame(xys, columns=['feature1'])
        data['original_cluster'] = labels
        data['new_cluster'] = new_labels

        self.data = data