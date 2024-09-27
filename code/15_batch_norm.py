"""
This is the exercise https://tensorgym.com/exercises/9
"""
import torch

def solution(data: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    This is the function implementation of the exercise.
    """
    numerator = data - torch.mean(data, dim=0, keepdim=True)
    denominator = torch.sqrt(torch.var(data, dim=0, keepdim=True) + epsilon)

    res = numerator / denominator
    return res


def equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    """
    Checks whether the tensors supplied are element-wise equal.
    """
    if x.shape != y.shape:
        return False
    
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            el1 = f'{x[row, col]:.4f}'
            el2 = f'{y[row, col]:.4f}'
            if el1 != el2:
                return False

    return True


def main():
    """
    This function runs the test cases.
    """
    cases = []

    # test case 1
    inputs = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]
    epsilon: float = 0.00001
    outputs = [
        [-1, -1],
        [0, 0],
        [1, 1],
    ]
    cases.append({
        'inputs': torch.tensor(inputs, dtype=torch.float),
        'epsilon': epsilon,
        'outputs': torch.tensor(outputs, dtype=torch.float),
    })

    # test case 2
    inputs = [
        [10, 4],
        [4, 2],
        [6, 0],
    ]
    epsilon: float = 0.00001
    outputs = [
        [1.0911, 1],
        [-0.8729, 0],
        [-0.2182, -1.0],
    ]
    cases.append({
        'inputs': torch.tensor(inputs, dtype=torch.float),
        'epsilon': epsilon,
        'outputs': torch.tensor(outputs, dtype=torch.float),
    })

    for ix, c in enumerate(cases):
        res = solution(c.get('inputs'), c.get('epsilon'))
        message = f'PASS: test case {ix + 1}' if equal(res, c.get('outputs')) else f'FAIL: test case {ix + 1}'
        print(message)


if __name__ == '__main__':
    main()