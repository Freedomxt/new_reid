import copy
import random
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import  Sampler


class RandomIdentifySampler(Sampler):
    def __init__(self,data_source,batch_size,num_instances):
        #super(RandomIdentifySampler, self).__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances

        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)

        for index, (_,pid,_) in enumerate(self.data_source):
            self.index_dic[pid].append(index)              #得到所有pid对应的样本的索引
        
        self.pids = list(self.index_dic.keys())
        
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length +=num - num % self.num_instances
    
    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs,size=self.num_instances,replace=True)
            
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

            avai_pids = copy.deepcopy(self.pids)
            final_pids = []

            while len(avai_pids) >= self.num_pids_per_batch:
                selected_pids = random.sample(avai_pids,self.num_pids_per_batch)
                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_pids.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)

            return iter(final_pids)

    def __len__(self):
        return self.length


            
            
            
