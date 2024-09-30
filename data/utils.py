import time
import datetime
import sklearn.preprocessing
import numpy as np
import pandas as pd
from typing import List
from torch import tensor, max, float32
from torchmetrics.functional import hamming_distance


def is_static(data: List) -> bool:
    return len(set(data)) == 1


def get_case_ids(tab):
    return list(set(tab["CaseID"]))


def translate_time(time_str) -> float:
    return time.mktime(
        datetime.datetime.strptime(time_str, "%Y/%m/%d %H:%M:%S").timetuple()
    )


def get_one_hot_encoder(dataset: pd.DataFrame, key: str):
    datas = dataset[key].unique()
    datas = datas.reshape([len(datas), 1])
    onehot = sklearn.preprocessing.OneHotEncoder()
    onehot.fit(datas)
    return onehot


def get_one_hot_encodings(
    onehot: sklearn.preprocessing.OneHotEncoder, datas: pd.Series
):
    return onehot.transform(datas.reshape(-1, 1)).toarray()


def get_node_features(dataset: pd.DataFrame, trace: pd.DataFrame) -> dict:
    columns_static = [c for c in trace if is_static(trace[c])]

    res = {}

    for key in trace:
        values = trace[key].values
        match key:
            case "Activity":
                onehot_activities = get_one_hot_encoder(dataset, "Activity")
                res[key] = tensor(get_one_hot_encodings(onehot_activities, values), dtype=float32)
            case "time:timestamp":
                res[key] = tensor(np.array(list(map(translate_time, values))), dtype=float32)
                res[key] = res[key].reshape(res[key].shape[0], 1)
            case "org:resource":
                if key not in columns_static:
                    res[key] = tensor(values, dtype=float32)
                    res[key] = res[key].reshape(res[key].shape[0], 1)
            case "lifecycle:transition":
                onehot_lifecyle_transition = get_one_hot_encoder(
                    dataset, "lifecycle:transition"
                )
                if key not in columns_static:
                    res[key] = tensor(
                        get_one_hot_encodings(onehot_lifecyle_transition, values), dtype=float32
                    )
                else:
                    res[key] = tensor(
                        get_one_hot_encodings(
                            onehot_lifecyle_transition, np.array([values[0]])
                        ), dtype=float32
                    )
            case "case:REG_DATE":
                if key not in columns_static:
                    res[key] = tensor(np.array(list(map(translate_time, values))), dtype=float32)
                else:
                    res[key] = tensor(
                        np.array(list(map(translate_time, np.array([values[0]])))), dtype=float32
                    )
                res[key] = res[key].reshape(res[key].shape[0], 1)
            case "case:AMOUNT_REQ":
                if key not in columns_static:
                    res[key] = tensor(values, dtype=float32)
                else:
                    res[key] = tensor([values[0]], dtype=float32)
                res[key] = res[key].reshape(res[key].shape[0], 1)

    return res


def compute_edges_indexs(node_features: dict, prefix_len):
    res = {}
    keys = node_features.keys()
    indexes = [[i, j] for i in range(prefix_len) for j in range(i + 1, prefix_len)]

    # activities indexes
    for k in keys:
        if len(node_features[k]) != 1:
            if k == "Activity":
                res[(k, "followed_by", k)] = indexes
                for k2 in keys:
                    if k2 != k:
                        if len(node_features[k2]) == 1:
                            res[(k, "related_to", k2)] = [
                                [i, 0] for i in range(prefix_len)
                            ]
                        else:
                            res[(k, "related_to", k2)] = [
                                [i, i] for i in range(prefix_len)
                            ]
            else:
                res[(k, "related_to", k)] = indexes

    return res


def compute_edges_features(node_features, edges_indexes):
    res = {}

    for k in edges_indexes:
        if k[0] == k[2]:
            indexes = edges_indexes[k]
            res[k] = []
            match k[0]:
                case "Activity":
                    for i in indexes:
                        res[k].append(
                            tensor(
                                [
                                    hamming_distance(
                                        node_features[k[0]][i[1]],
                                        node_features[k[0]][i[0]],
                                        task="binary",
                                    )
                                ]
                            )
                        )
                case "org:resource":
                    for i in indexes:
                        res[k].append(
                            tensor(
                                [node_features[k[0]][i[1]] - node_features[k[0]][i[0]]]
                            )
                        )
                case "time:timestamp":
                    for i in indexes:
                        res[k].append(
                            tensor(
                                [node_features[k[0]][i[1]] - node_features[k[0]][i[0]]]
                            )
                        )

    return res
