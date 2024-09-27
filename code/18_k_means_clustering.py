"""
This file implements the solution for the exercise https://tensorgym.com/exercises/12
"""
import torch
import numpy as np

def k_means_clustering(data: np.ndarray, k: int) -> np.ndarray:
    """
    The implementation for the solution to this exercise.
    """
    data = torch.from_numpy(data)

    rand_ix = torch.randperm(data.shape[0])[0:k]
    centroids = data[rand_ix]
    clusters = []

    for row in range(data.shape[0]):
        target_ix = -1
        min_distance = float('inf')
        row_data = data[row]

        for ix, c in enumerate(centroids):
            distance = torch.norm(row_data - c).item()
            target_ix = ix if distance < min_distance else target_ix
            min_distance = min(min_distance, distance)
        
        # assign the row to a cluster
        if len(clusters) < (target_ix + 1):
            clusters.append([row_data.numpy().tolist()])
        else:
            clusters[target_ix].append(row_data.numpy().tolist())

    print(clusters)

def main():
    """
    The main function for running the test cases.
    """
    inputs = [
        [1, 2],
        [5, 5],
        [1, 5],
        [8, 8.5],
    ]
    inputs = torch.tensor(inputs, dtype=torch.float)
    k: int = 2
    inputs = inputs.numpy()

    res = k_means_clustering(inputs, k)
    print(res)


if __name__ == '__main__':
    main()
