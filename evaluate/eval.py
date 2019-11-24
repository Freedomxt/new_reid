import torch
import torchvision
import numpy as np
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Evaluator(object):
    def __init__(self,model,val_loader,num_query):
        self.loader = val_loader
        self.num_query = num_query
        self.model = model
        self.feats = []
        self.pids = []
        self.camids = []

    def extract_feature(self):
        #self.model.to(device)
        self.model.to(device)
        self.model.eval()
        
        with torch.no_grad():
            for data in tqdm(self.loader):
                img,pid,camid = data
                img = img.to(device)
                _,feat = self.model(img)
                self.feats.append(feat)
                self.pids.extend(np.asarray(pid))
                self.camids.extend(np.asarray(camid))
            self.feats = torch.cat(self.feats,dim=0)
            
    def compute_dists(self):
        #get the query informations
        qf = self.feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        
        #get the gallery informations
        gf = self.feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        
        m,n = qf.shape[0],gf.shape[0]
        distmat = torch.pow(qf,2).sum(dim=1,keepdim=True).expand(m,n) + \
                  torch.pow(gf,2).sum(dim=1,keepdim=True).expand(n,m).t()
        distmat.addmm_(1,-2,qf,gf.t())
        distmat = distmat.cpu().numpy()
        cmc,mAP = self.eval_func(distmat,q_pids,g_pids,q_camids,g_camids)
        return cmc,mAP


    def eval_func(self,distmat,q_pids,g_pids,q_camids,g_camids,max_rank=50):
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, only {}".format(num_g))
        indices = np.argsort(distmat,axis=1)
        matches = (g_pids[indices] == q_pids[:,np.newaxis]).astype(np.int32)

        all_cmc = []
        all_AP = []
        num_valid_q = 0.
        for q_idx in range(num_q):
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]
            
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                continue

            cmc = orig_cmc.cumsum()   #累加
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()

            tmp_cmc = [x/(i+1.) for i,x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc

            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q

        mAP = np.mean(all_AP)

        return all_cmc,mAP

    def run(self):
        self.extract_feature()
        cmc,mAP = self.compute_dists()
        return cmc,mAP















            


        
        





        
        
        




                
                
                
                
                



