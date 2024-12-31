import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
from typing import Dict


class Location:
    def __init__(self,  info:Dict[str, Dict]):
        self.model = KMeans(n_clusters=5)
        
        
    def __call__(self, info:Dict[str, Dict]):
        xs = [[item["xywh"][2]] for _, item in info.items()]
        
        _ = self.model.fit_predict(xs)
        cluster_centers = self.model.cluster_centers_
        flat_arr = cluster_centers.flatten()
        sorted_indices = np.argsort(flat_arr)
        self.indic_mapping =  {sorted_indices[i]:i  for i in range(len(flat_arr))}
    
        self.rows = [[],[],[], [],[]]
        
        
        for _, item in info.items():
            x = [item["xywh"][2]]
            clust = int(self.model.predict(x)[0])
            item["row"] = self.indic_mapping[clust]
            self.rows[item["row"]].append(item)
        

        def get_row_value(sub_list):
            if sub_list:
                return sub_list[0].get("row")
            return None
        
        self.rows.sort(key=get_row_value)
        for row in self.rows:
            row.sort(key=lambda x: x["xywh"][0])
            for col_idx, item in enumerate(row):
                item["col"] = col_idx
                # 将更新后的 item 写回 info 字典
                for name, info_item in info.items():
                    item["name"] = name
                    if np.array_equal(info_item["xywh"], item["xywh"]):
                        info[name] = item
        return info, self.rows