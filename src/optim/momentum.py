import torch.optim as optim


class Momentum(optim.SGD):
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0):
        super(Momentum, self).__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

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
