"""
This implements the solution for the exercise https://tensorgym.com/exercises/15
"""
import torch


class CustomReLUFn(torch.autograd.Function):
    """
    The custom ReLU function.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass.
        """
        ctx.save_for_backward(x)

        res = x.masked_fill(x <= 0, 0)
        return res
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Implements the backward pass.
        """
        inputs, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        dx = inputs.masked_fill(inputs > 0, 1)
        dx = dx.masked_fill(dx < 1, 0)

        grad_input = grad_output * dx

        return grad_input

def compute_custom_relu(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ReLU using the custom ReLU implementation.
    """
    relu = CustomReLUFn.apply
    inputs = inputs.clone().detach().requires_grad_(True)
    outputs = relu(inputs)
    outputs.backward(torch.ones_like(inputs))
    return outputs, inputs.grad


def main():
    inputs = [-1, -0.5, 1]
    inputs = torch.tensor(inputs, dtype=torch.float)

    outputs, inputs_grad = compute_custom_relu(inputs)

    print(outputs)
    print(inputs_grad)


if __name__ == '__main__':
    main()
