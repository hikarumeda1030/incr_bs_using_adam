import torch
import torch.optim as optim


class NHSB(optim.Optimizer):
    def __init__(self, params, lr=0.1, beta=0.9):
        defaults = dict(lr=lr, beta=beta)
        super(NHSB, self).__init__(params, defaults)

    def step(self, closure=None, iteration=0):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        gradient_list = []
        for group in self.param_groups:
            beta = group['beta']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)

                momentum = state['momentum']

                # Update momentum
                momentum.mul_(beta).add_(1 - beta, grad)

                # Update parameters
                p.data.add_(-lr, momentum)

                if iteration != 0:
                    gradient_list.append(grad.clone())

        if iteration != 0:
            return gradient_list

        return loss
