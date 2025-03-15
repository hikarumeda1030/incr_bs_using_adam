import torch.optim as optim


class Adam(optim.Adam):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(Adam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

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
