import torch
from torch import nn


def normalize(x, axis=-1):
    
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def hard_example_mining(dist_mat,labels,return_inds=False):

    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    is_pos = labels.expand(N,N).eq(labels.expand(N,N).t())
    is_neg = labels.expand(N,N).ne(labels.expand(N,N).t())

    dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N,-1),1,keepdim=True)
    dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N,-1),1,keepdim=True)
    
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap,dist_an



class TripletLoss(object):

    def __init__(self,margin=None):
        super(TripletLoss, self).__init__()
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat,labels,normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat,axis=-1)

        dist_mat = euclidean_dist(global_feat,global_feat)
        dist_ap,dist_an = hard_example_mining(dist_mat,labels)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin is not None:
            loss = self.ranking_loss(dist_an,dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss,dist_ap,dist_an
        