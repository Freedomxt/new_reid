import torch.nn.functional as F
from .soft_loss import getSoftLabel
from .triplet_loss import TripletLoss

def make_loss(args):
    sampler = args.sampler
    #triplet = TripletLoss(args.margin_tri)
    if sampler =='softmax':
        def loss_func(score,target,soft_label=None):
            return F.cross_entropy(score,target)
    elif sampler == 'triplet':
        pass
    elif sampler == 'soft_label':
        def loss_func(score,label,soft_label):
            T = 10
            soft_label = getSoftLabel(soft_label,T)
            prob = getSoftLabel(score,T)
            KLloss = (T**2)*F.kl_div(prob.log(),soft_label,reduction='batchmean')
            softmax_loss = F.cross_entropy(score,label)
            return KLloss + softmax_loss,KLloss,softmax_loss

    return loss_func