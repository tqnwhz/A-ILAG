import numpy as np
import math
from collections import defaultdict
import torch
import random
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def linear_assignment(scores,
                      group_capacity=32,
                      times=1,
                      type='random',
                      k=200):
    N = len(scores)
    n_groups = math.ceil(N / group_capacity)
    scores = torch.tensor(scores, device='cpu')
    best_difficulty = 0
    best_clusters = []
    for i in range(times):
        if type == 'random':
            centers = random.sample(range(N), n_groups)
        elif type == 'greedy-random':
            init = random.randint(0, N - 1)
            centers = [init]
            scores_copy = scores.clone()
            for i in range(n_groups - 1):
                prev_center = centers[-1]
                scores_copy[:, prev_center] = 2
                center_socres = scores_copy.index_select(
                    dim=0, index=torch.tensor(centers)).max(0).values
                candidate_centers = center_socres.topk(k=k,
                                                       dim=-1,
                                                       largest=False).indices
                centers.append(random.choice(candidate_centers).item())
        scores_copy = scores.clone()
        eye = torch.eye(N, dtype=torch.bool)
        scores_copy.masked_fill_(eye, 0)
        center_socres = scores_copy.index_select(dim=0,
                                                 index=torch.tensor(centers))
        clusters = [[x] for x in centers]
        rest = set(range(N)) - set(centers)
        while len(rest) > 0:
            subset = random.sample(rest, min(len(centers), len(rest)))
            subset.sort()
            rest = rest - set(subset)
            similarity = center_socres.index_select(dim=1,
                                                    index=torch.tensor(subset))
            row_ind, col_ind = linear_sum_assignment(similarity.numpy(),
                                                     maximize=True)
            col_ind = [subset[i] for i in col_ind]
            for row, col in zip(row_ind, col_ind):
                clusters[row].append(col)

            if len(subset) == group_capacity:
                center_socres = center_socres + scores_copy.index_select(
                    dim=0, index=torch.tensor(col_ind))
        difficulty = []
        for c in clusters:
            index = torch.tensor(c)
            d = scores_copy.index_select(dim=0, index=index).index_select(
                dim=1, index=index).sum(1).mean() / (group_capacity - 1)
            difficulty.append(d)
        difficulty = np.mean(difficulty)
        if difficulty > best_difficulty:
            best_difficulty = difficulty
            best_clusters = clusters

    return best_clusters, best_difficulty


def sparse_linear_assignment(scores: csr_matrix,
                             group_capacity=32,
                             times=1,
                             type='random'):
    N = scores.shape[0]
    n_groups = math.ceil(N / group_capacity)
    best_difficulty = 0
    best_clusters = []
    for i in range(times):
        centers = random.sample(range(N), n_groups)
        center_socres = scores[centers]
        clusters = [[x] for x in centers]
        rest = set(range(N)) - set(centers)
        while len(rest) > 0:
            subset = random.sample(rest, min(len(centers), len(rest)))
            subset.sort()
            rest = rest - set(subset)
            similarity = center_socres[:, subset]
            row_ind, col_ind = linear_sum_assignment(similarity.toarray(),
                                                     maximize=True)
            col_ind = [subset[i] for i in col_ind]
            for row, col in zip(row_ind, col_ind):
                clusters[row].append(col)

            if len(subset) == group_capacity:
                center_socres = center_socres + scores[col_ind]
        difficulty = []
        for c in clusters:
            index = torch.tensor(c)
            d = scores[index].tocsc()[:, index].toarray().sum(1).mean()
            difficulty.append(d)
        difficulty = np.mean(difficulty)
        if difficulty >= best_difficulty:
            best_difficulty = difficulty
            best_clusters = clusters

    return best_clusters, best_difficulty
