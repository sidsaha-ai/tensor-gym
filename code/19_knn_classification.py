"""
This contains the solution to the exercise https://tensorgym.com/exercises/11.
"""
import numpy as np


def knn_classifier(
        train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray, k: int,
) -> np.ndarray:
    """
    This function is the implementation of this exercise.
    """
    test_data = test_data[:, np.newaxis, :]

    distances = np.sqrt(np.sum(
        np.power(test_data - train_data, 2), axis=-1,
    ))
    sorted_ix = np.argsort(distances, axis=-1)[:, 0:k]
    labels = train_labels[sorted_ix]
    bin_counts = np.apply_along_axis(np.bincount, axis=-1, arr=labels, minlength=np.max(train_labels))
    predictions = np.argmax(bin_counts, axis=1)

    return predictions


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

    # test case 2
    train_data = [
        [1, 1.5],
        [2, 2.5],
        [3, 3.5],
    ]
    train_labels = [0, 1, 0]
    test_data = [
        [1.5, 2],
        [3, 3],
    ]
    k: int = 2
    outputs = [0, 0]
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
        message = f'PASS: test case {ix + 1}' if (res == c.get('outputs')).all() else f'FAIL: test case {ix + 1}'
        print(message)


if __name__ == '__main__':
    main()
