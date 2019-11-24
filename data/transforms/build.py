import torchvision.transforms as T
from .transforms import RandomErasing

def build_transforms(args,is_train=True):
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    if is_train:
        transform = T.Compose([
            T.Resize(args.image_size),
            T.RandomHorizontalFlip(p=args.flip_p),
            T.Pad(args.input_pading),
            T.RandomCrop(args.image_size),
            T.ToTensor(),
            normalize_transform,
            #RandomErasing(probability=0.5,mean=[0.485, 0.456, 0.406])

        ])
    else:
        transform = T.Compose([
            T.Resize(args.image_size),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
