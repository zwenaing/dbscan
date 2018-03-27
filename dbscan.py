from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd

class DBSCAN:

    def __init__(self, epsilon=0.5, min_samples=5):
        self.epsilon = epsilon
        self.min_samples = min_samples

    def core_border_classification(self, no_ngbrs_dict, ngbrs_dict):
        cls_dict = {}
        for k, v in no_ngbrs_dict.items():
            if v >= self.min_samples:
                cls_dict[k] = "core"      # classify the core point N(p) >= threshold
            elif v > 0:
                cls_dict[k] = "border"    # classify the border point N(p) < threshold
            # noise points are removed if they have no neighbors
        delete_list = []
        # remove border points if their neighbors are all border points
        for k, v in cls_dict.items():
            if v == "border":   # if k is a border point
                one_core = False
                # check if one of the neighbors is core
                for ngbr in ngbrs_dict[k]:
                    if cls_dict[ngbr] == "core":
                        one_core = True
                # if none of the neighbors is core, mark it as noise point and remove it
                if not one_core:
                    delete_list.append(k)
        for i in delete_list:
            del cls_dict[i]
        return cls_dict

    def get_neighbors(self, data_X, metric):
        ngbrs_dict = {}
        for i, x1 in enumerate(data_X):
            ngbrs_dict[i] = []
            x1 = x1.reshape(1, -1)
            for j, x2 in enumerate(data_X):
                x2 = x2.reshape(1, -1)
                dist = pairwise_distances(x1, x2, metric)       # calculate the distance
                # if the distance < epsilon, mark as neighbor
                if dist < self.epsilon:
                    ngbrs_dict[i].append(j)
        return ngbrs_dict

    def get_number_of_neighbors(self, ngbrs_dict):
        """
        Get the number of neighbors for each datapoint.
        :param ngbrs_dict: the neighbors for each datapoint.
        :return: the number of neighbors for each datapoint.
        """
        no_ngbrs_dict = {}
        for k, v in ngbrs_dict.items():
            no_ngbrs_dict[k] = len(v)
        return no_ngbrs_dict

    def breadth_first_search(self, graph):
        discovered = {}
        cluster_no = 0
        nodes = set(graph.keys())
        while nodes:
            visited = self.breadth_first_search_helper(graph, nodes.pop())
            if visited:
                discovered[cluster_no] = visited
                cluster_no += 1
                nodes = nodes - visited
        return discovered

    def breadth_first_search_helper(self, graph, start):
        visited, queue = set(), [start]
        while queue:
            current = queue.pop(0)
            for n in graph[current]:
                if n not in visited:
                    queue.append(n)
                    visited.add(n)
        return visited

    def assign_core_clusters(self, cls_dict, ngbrs_dict):
        core_pts_graph = {}
        for i in cls_dict.keys():
            core_pts_graph[i] = []
        for i, ngbrs in ngbrs_dict.items():
            if i in cls_dict and cls_dict[i] == "core":
                for ngbr in ngbrs:
                    if cls_dict[ngbr] == "core":
                        if ngbr not in core_pts_graph[i] and ngbr != i:
                            core_pts_graph[i].append(ngbr)
                        if i not in core_pts_graph[ngbr] and ngbr != i:
                            core_pts_graph[ngbr].append(i)
        core_clusters = self.breadth_first_search(core_pts_graph)
        no_of_clusters = len(core_clusters)
        core_assignment = {}
        for i, cluster in core_clusters.items():
            for pt in cluster:
                core_assignment[pt] = i
        return no_of_clusters, core_assignment

    def assign_border_clusters(self, cls_dict, ngbrs_dict, core_assignment):
        border_assignment = {}
        for point in cls_dict.keys():
            if cls_dict[point] == "border":
                ngbrs = ngbrs_dict[point]
                if ngbrs:
                    for ngbr in ngbrs:
                        if ngbr == "core":
                            border_assignment[point] = core_assignment[ngbr]
        return border_assignment

    def fit(self, data_X, metric="euclidean"):
        ngbrs_dict = self.get_neighbors(data_X, metric)     # get the neighbors of each point
        no_ngbrs_dict = self.get_number_of_neighbors(ngbrs_dict)    # get the number of neighbors of each point
        cls_dict = self.core_border_classification(no_ngbrs_dict, ngbrs_dict)    # classify core and border datapoints
        no_of_clusters, core_assignment = self.assign_core_clusters(cls_dict, ngbrs_dict)     # assign core points to clusters
        border_assignment = self.assign_border_clusters(cls_dict, ngbrs_dict, core_assignment)    # assign border points to clusters
        # combine core and border assignments
        for k, v in border_assignment.items():
            core_assignment[k] = v
        result = {"no_of_clusters": no_of_clusters, "labels": core_assignment}
        return result

if __name__ == "__main__":
    data = pd.read_csv('dbscan.csv')
    data_X = np.vstack((data['x'], data['y']))
    data_X = data_X.transpose()
    dbscan = DBSCAN(7.5, 3)
    result = dbscan.fit(data_X, "euclidean")

    print(result['no_of_clusters'])
    print(result['labels'])