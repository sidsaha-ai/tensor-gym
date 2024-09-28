"""
This contains the solution to the exercise https://tensorgym.com/exercises/11.
"""
import numpy as np

from collections import Counter

def euclid_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Finds the distance between x and y.
    """
    return np.sqrt(np.sum(np.pow(x - y, 2)).item())

def knn_classifier(
        train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray, k: int,
) -> np.ndarray:
    """
    This function is the implementation of this exercise.
    """
    predictions = np.zeros(test_data.shape[0])  # the number of test samples to be predicted

    for row, tst_data in enumerate(test_data):
        distances = np.zeros(train_data.shape[0])  # the distance of this test sample with every train sample

        for ix, trn_data in enumerate(train_data):
            distances[ix] = euclid_distance(tst_data, trn_data)

        sorted_ix = np.argsort(distances)[0:k]  # get the nearest K
        label_freq = Counter(train_labels[ix].item() for ix in sorted_ix)
        print(label_freq)

def main():
    """
    This method will run the test cases.
    """
    cases = []

    # test case 1
    train_data = [
        [1, 2],
        [2, 3],
        [3, 4],
    ]
    train_labels = [0, 1, 0]
    test_data = [
        [1.5, 2.5],
    ]
    k: int = 2
    outputs = [0]
    cases.append({
        'train_data': np.array(train_data, dtype=float),
        'train_labels': np.array(train_labels, dtype=int),
        'test_data': np.array(test_data, dtype=float),
        'k': k,
        'outputs': np.array(outputs, int),
    })

    for ix, c in enumerate(cases):
        res = knn_classifier(
            c.get('train_data'), c.get('train_labels'), c.get('test_data'), c.get('k'),
        )
        print(res)


if __name__ == '__main__':
    main()
