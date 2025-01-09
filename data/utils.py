import time
import datetime
import sklearn.preprocessing
import numpy as np
import pandas as pd
from typing import List
from torch import cat, tensor, max, float32
from torchmetrics.functional import hamming_distance, pairwise_euclidean_distance
from sklearn.feature_extraction import FeatureHasher
from torch.nn.functional import cosine_similarity

import torch 

def is_static(data: List) -> bool:
    return len(set(data)) == 1


def get_case_ids(tab):
    return list(tab["CaseID"].unique())

from math import log

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
                res[key] = tensor(
                    get_one_hot_encodings(onehot_activities, values), dtype=float32
                )
            case "time:timestamp":
                res[key] = tensor(
                    np.array(list(map(translate_time, values))), dtype=float32
                )
                res[key] = res[key].reshape(res[key].shape[0], 1)
            case "org:resource":
                onehot_resource = get_one_hot_encoder(dataset, "org:resource")
                # if key not in columns_static:
                res[key] = tensor(
                    get_one_hot_encodings(onehot_resource, values),
                    dtype=float32,
                )
                # )
                # tensor(
                #    get_one_hot_encodings(onehot_resource, values),
                #    #resourse_featHash.transform([[str(x)] for x in values]).toarray(),
                #    dtype=float32,
                # ),

                # else:
                #    res[key] = (tensor(
                #            resourse_featHash.transform([[str(values[0])]]).toarray(), dtype=float32
                #        ),tensor(
                #        get_one_hot_encodings(onehot_resource, np.array([values[0]])),
                #        dtype=float32,
                #    ))
                #    # res[key][1] = res[key][1].reshape(res[key][1].shape[0], 1)

            case "lifecycle:transition":
                onehot_lifecyle_transition = get_one_hot_encoder(
                    dataset, "lifecycle:transition"
                )
                if key not in columns_static:
                    res[key] = tensor(
                        get_one_hot_encodings(onehot_lifecyle_transition, values),
                        dtype=float32,
                    )
                else:
                    res[key] = tensor(
                        get_one_hot_encodings(
                            onehot_lifecyle_transition, np.array([values[0]])
                        ),
                        dtype=float32,
                    )
            case "case:REG_DATE":
                if key not in columns_static:
                    res[key] = tensor(
                        np.array(list(map(translate_time, values))), dtype=float32
                    )
                else:
                    res[key] = tensor(
                        np.array(list(map(translate_time, np.array([values[0]])))),
                        dtype=float32,
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
                            #cat( (tensor([cosine_similarity(node_features[k[0]][i[1]],node_features[k[0]][i[0]], dim=0)]),tensor([i[1] - i[0]])) )
                            cat(( node_features[k[0]][i[1]]- node_features[k[0]][i[0]],tensor([i[1] - i[0]]) ))
                        )
                case "org:resource":
                    for i in indexes:
                        res[k].append(
                            #cat( (tensor([cosine_similarity(node_features[k[0]][i[1]], node_features[k[0]][i[0]], dim=0)]),tensor([i[1] - i[0]])) )
                            cat(( node_features[k[0]][i[1]]- node_features[k[0]][i[0]],tensor([i[1] - i[0]]) ))

                        )
                case "time:timestamp":
                    for i in indexes:
                        res[k].append(
                            tensor(
                                [
                                    node_features[k[0]][i[1]]
                                    - node_features[k[0]][i[0]],
                                    i[1] - i[0],
                                ]
                            )
                        )

    return res


from scipy.stats import pearsonr
import networkx as nx

from operator import itemgetter
import pandas as pd


class ResourcePoolAnalyser():
    """
        This class evaluates the tasks durations and associates resources to it
     """

    def __init__(self, log, drawing=False, sim_threshold=0.7):
        """constructor"""
        self.data = self.read_resource_pool(log)
        self.drawing = drawing
        self.sim_threshold = sim_threshold
        
        self.tasks = {val: i for i, val in enumerate(self.data["Activity"].unique())}
        self.users = {val: i for i, val in enumerate(self.data["org:resource"].unique())}
        
        self.roles, self.resource_table = self.discover_roles()

    def read_resource_pool(self, log):
        if isinstance(log, pd.DataFrame):
            filtered_list = log[['Activity', 'org:resource']]
        else:
            filtered_list = pd.DataFrame(log.data)[['Activity', 'org:resource']]
        #filtered_list = filtered_list[~filtered_list.task.isin(['Start', 'End'])]
        #filtered_list = filtered_list[filtered_list.user != 'AUTO']
        return filtered_list


    def discover_roles(self):
        associations = lambda x: (self.tasks[x['Activity']], self.users[x['org:resource']])
        self.data['ac_rl'] = self.data.apply(associations, axis=1)
    
        freq_matrix = (self.data.groupby(by='ac_rl')['Activity']
                       .count()
                       .reset_index()
                       .rename(columns={'Activity': 'freq'}))
        freq_matrix = {x['ac_rl']: x['freq'] for x in freq_matrix.to_dict('records')}
        
        profiles = self.build_profile(freq_matrix)
    
        print(((20 / 100)* 100),'Analysing resource pool ')
        # building of a correl matrix between resouces profiles
        correl_matrix = self.det_correl_matrix(profiles)
        print(((40 / 100)* 100),'Analysing resource pool ')
        # creation of a rel network between resouces
        g = nx.Graph()
        for user in self.users.values():
            g.add_node(user)
        for rel in correl_matrix:
            # creation of edges between nodes excluding the same elements
            # and those below the similarity threshold 
            if rel['distance'] > self.sim_threshold and rel['x'] != rel['y']:
                g.add_edge(rel['x'],
                           rel['y'],
                           weight=rel['distance'])
        print(((60 / 100) * 100),'Analysing resource pool ')
        # extraction of fully conected subgraphs as roles
        sub_graphs = list(nx.connected_components(g))
        print(((80 / 100) * 100),'Analysing resource pool ')
        # role definition from graph
        roles = self.role_definition(sub_graphs)
        # plot creation (optional)
        # if drawing == True:
        #     graph_network(g, sub_graphs)
        print(((100 / 100)* 100),'Analysing resource pool ')
        
        
        import pprint
        pprint.pprint(f"ROLES \n\n{roles}\n\n")
        return roles
    
    def build_profile(self, freq_matrix):
        profiles=list()
        for user, idx in self.users.items():
            profile = [0,] * len(self.tasks)
            for ac_rl, freq in freq_matrix.items():
                if idx == ac_rl[1]:
                    profile[ac_rl[0]] = freq
            profiles.append({'user': idx, 'profile': profile})
        return profiles



    def det_correl_matrix(self, profiles):
        correl_matrix = list()
        import numpy
        for profile_x in profiles:
            for profile_y in profiles:
                x = numpy.array(profile_x['profile'])
                y = numpy.array(profile_y['profile'])
                r_row, p_value = pearsonr(x, y)
                correl_matrix.append(({'x': profile_x['user'],
                                            'y': profile_y['user'],
                                            'distance': r_row}))
        return correl_matrix

    def role_definition(self, sub_graphs):
        user_index = {v: k for k, v in self.users.items()}
        records= list()
        for i in range(0, len(sub_graphs)):
            users_names = [user_index[x] for x in sub_graphs[i]]
            records.append({'role': 'Role '+ str(i + 1),
                            'quantity': len(sub_graphs[i]),
                            'members': users_names})
        #Sort roles by number of resources
        records = sorted(records, key=itemgetter('quantity'), reverse=True)
        for i in range(0,len(records)):
            records[i]['role']='Role '+ str(i + 1)
        resource_table = list()
        for record in records:
            for member in record['members']:
                resource_table.append({'role': record['role'],
                                       'resource': member})
        return records, resource_table
    
    
def get_resource_role_map(log):
    r = ResourcePoolAnalyser(log)
    return r.resource_table