
import torch

def make_optimizer(args,model):
    params = []
    for key,value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.base_learning_rate                #0.00035
        weight_decay = args.weight_decay             #0.0005
        if "bias" in key:
            lr = args.base_learning_rate
            weight_decay = 0.0005
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.Adam(params)
    return optimizer