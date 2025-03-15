from optim.sgd import SGD
from optim.adam import Adam
from optim.adamw import AdamW
from optim.rmsprop import RMSprop
from optim.momentum import Momentum
from optim.nhsb_momentum import NHSB


def setup_optimizer(optimizer_name, parameters, lr, **kwargs):
    """
    Returns an optimizer instance based on the provided name and parameters.

    Args:
        optimizer_name (str): Name of the optimizer to use.
        parameters (iterable): Model parameters to optimize.
        lr (float): Learning rate for the optimizer.
        kwargs: Additional keyword arguments for specific optimizers.

    Returns:
        torch.optim.Optimizer: The optimizer instance.
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "sgd":
        return SGD(parameters, lr=lr, **kwargs)
    elif optimizer_name == "adam":
        return Adam(parameters, lr=lr, **kwargs)
    elif optimizer_name == "adamw":
        return AdamW(parameters, lr=lr, **kwargs)
    elif optimizer_name == "rmsprop":
        return RMSprop(parameters, lr=lr, **kwargs)
    elif optimizer_name == "momentum":
        return Momentum(parameters, lr=lr, **kwargs)
    elif optimizer_name == "custom_momentum":
        return NHSB(parameters, lr=lr, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported optimizers are: sgd, adam, adamw, rmsprop, momentum.")
