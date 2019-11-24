from torch.utils.data import DataLoader
from .transforms import build_transforms
from .datasets import init_dataset,ImageDataset,ImageDatasetSoft
from .samplers import RandomIdentifySampler
import torch
import torchvision.transforms as T

def train_collate_fn_soft(batch):
    imgs, pids, _, _, soft_labels= zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    soft_labels = torch.tensor(soft_labels)
    return torch.stack(imgs, dim=0), pids, soft_labels

def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

def make_data_loader(args):
    train_transforms = build_transforms(args,is_train=True)
    val_transforms = build_transforms(args,is_train=False)
    
    dataset = init_dataset(args.dataset)


    train_set = ImageDataset(dataset.train,train_transforms)
    if args.sampler == 'softmax':
        train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=8,collate_fn=train_collate_fn)
    else:
        train_loader = DataLoader(train_set,batch_size=args.batch_size,
                                  sampler=RandomIdentifySampler(dataset.train,args.batch_size,4),
                                  num_workers=8,collate_fn=train_collate_fn)


    val_set = ImageDataset(dataset.query + dataset.gallery,val_transforms)
    val_loader = DataLoader(val_set,batch_size=256,shuffle=False,num_workers=8,collate_fn=val_collate_fn)
    

    return train_loader,val_loader,len(dataset.query),dataset.num_train_pids,len(dataset.train)

def soft_label_loader(args):
    transform = T.Compose([
        T.Resize(args.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = init_dataset(args.dataset)

    train_set = ImageDataset(dataset.train,transform)
    train_loader = DataLoader(train_set,batch_size=256,shuffle=False,num_workers=8,collate_fn=train_collate_fn)

    return train_loader,dataset.num_train_pids,dataset.train

def soft_train_loader(args,dataset,root):
    train_transforms = build_transforms(args, is_train=True)
    train_set = ImageDatasetSoft(dataset,train_transforms,root)
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=8,collate_fn=train_collate_fn_soft)

    return train_loader
    




