"""
This is the solution to the exercise https://tensorgym.com/exercises/16
"""
import torch

def mse_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    This function implements the solution to this exercise.
    """
    res = torch.sum(torch.pow(inputs - targets, 2)) / inputs.numel()
    return res

def equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    x_str: str = f'{x.item():.4f}'
    y_str: str = f'{y.item():.4f}'
    if x_str != y_str:
        return False
    return True


def main():
    """
    This function runs the test cases.
    """
    cases = []

    mse_loss = torch.nn.MSELoss()

    # test case 1
    inputs = [1, 2, 3]
    targets = [1, 2.5, 2.5]
    inputs = torch.tensor(inputs, dtype=torch.float)
    targets = torch.tensor(targets, dtype=torch.float)
    cases.append({
        'inputs': inputs,
        'targets': targets,
        'outputs': mse_loss(inputs, targets),
    })

    # test case 2
    inputs = [
        [3, -3, 0],
        [1, 0, 0],
    ]
    targets = [
        [3, -2.5, 0.5],
        [1, -1, 1],
    ]
    inputs = torch.tensor(inputs, dtype=torch.float)
    targets = torch.tensor(targets, dtype=torch.float)
    cases.append({
        'inputs': inputs,
        'targets': targets,
        'outputs': mse_loss(inputs, targets),
    })

    for ix, c in enumerate(cases):
        res = mse_loss(c.get('inputs'), c.get('targets'))
        message = f'PASS: test case {ix + 1}' if equal(res, c.get('outputs')) else f'FAIL: test case {ix + 1}'
        print(message)


if __name__ == '__main__':
    main()
