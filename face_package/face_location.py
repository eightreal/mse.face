import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import sys
from typing import Dict
from loguru import logger

class Location:
    def __init__(self,):
        # 聚类模型
        self.model = KMeans(n_clusters=5)
        
        
    def __call__(self, info:Dict[str, Dict]):
        xs = [[item["xywh"][1]] for _, item in info.items()]
        
        _ = self.model.fit_predict(xs)
        cluster_centers = self.model.cluster_centers_
        flat_arr = cluster_centers.flatten()
        sorted_indices = np.argsort(flat_arr)[::-1]
        
        logger.info(sorted_indices)
        
        self.indic_mapping =  {sorted_indices[i]:i  for i in range(len(flat_arr))}
    
        self.rows = [[],[],[], [],[]]

        for _, item in info.items():
            x = [[item["xywh"][1]]]
            clust = int(self.model.predict(x)[0])
            item["row"] = self.indic_mapping[clust] + 1
            self.rows[item["row"] -1].append(item)
            
        logger.info(f"row1 {len(self.rows[0])}, row2 {len(self.rows[1])}, row3 {len(self.rows[2])}, row4 {len(self.rows[3])}, row5 {len(self.rows[4])}")
        def get_row_value(sub_list):
            if sub_list:
                return sub_list[0].get("row")
            return None
        
        self.rows.sort(key=get_row_value)
        for row in self.rows:
            row.sort(key=lambda x: x["xywh"][0])
            for col_idx, item in enumerate(row):
                item["col"] = col_idx + 1
                # 将更新后的 item 写回 info 字典
                for name, info_item in info.items():
                    if np.array_equal(info_item["xywh"], item["xywh"]):
                        item["name"] = name
                        info[name] = item
        return info, self.rows