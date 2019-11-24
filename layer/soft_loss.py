import torch.nn.functional as F

def getSoftLabel(logits,T=10):
    softLabel = F.softmax(logits/T,dim=1)
    return softLabel


