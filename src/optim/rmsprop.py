import torch.optim as optim


class RMSprop(optim.RMSprop):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super(RMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)

    def step(self, closure=None, iteration=0):
        loss = super().step(closure)

        if iteration != 0:
            gradient_list = []
            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        gradient_list.append(param.grad.data.clone())
            return gradient_list
        return loss
