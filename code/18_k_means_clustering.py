"""
This file implements the solution for the exercise https://tensorgym.com/exercises/12
"""
import numpy as np


def k_means_clustering(data: np.ndarray, k: int) -> np.ndarray:
    """
    The implementation for the solution to this exercise.
    """
    np.random.seed(0)

    # init
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    max_iter: int = 2
    tolerance: float = 1e-4

    for _ in range(max_iter):
        old_centroids = centroids

        # assign
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        centroids = np.vstack([data[labels == i].mean(axis=0) for i in range(k)])

        # check for convergence
        if np.all(np.abs(centroids - old_centroids) < tolerance):
            break

    return labels.astype(int)


def main():
    """
    The main function for running the test cases.
    """
    cases = []

    # test case 1
    inputs = [
        [1, 2],
        [5, 5],
        [1, 5],
        [8, 8.5],
    ]
    k: int = 2
    outputs = [0, 0, 0, 1]
    cases.append({
        'inputs': np.array(inputs, dtype=float),
        'k': k,
        'outputs': np.array(outputs, dtype=float),
    })

    # test case 2
    inputs = [
        [1, 1],
        [2, 2],
        [9, 9],
        [8, 8],
    ]
    k: int = 2
    outputs = [1, 1, 0, 0]
    cases.append({
        'inputs': np.array(inputs, dtype=float),
        'k': k,
        'outputs': np.array(outputs, dtype=float),
    })

    for ix, c in enumerate(cases):
        res = k_means_clustering(c.get('inputs'), c.get('k'))
        message = f'PASS: test case {ix + 1}' if np.allclose(res, c.get('outputs'), atol=1e-4) else f'FAIL: test case {ix + 1}'
        print(message)


if __name__ == '__main__':
    main()
