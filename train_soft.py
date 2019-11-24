import argparse
import os
import sys
import torch
import torchvision
from tqdm import tqdm
import pickle

from data import make_data_loader,soft_train_loader
from model import build_model
from solver import make_optimizer, WarmupMultiStepLR
from layer import make_loss
from evaluate import Evaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet50(pretrained=True)


def save_model(args, model, optimizer, epoch):
    saved_path = os.path.join(args.model_path, args.model_name)
    if args.sampler=='soft_label':
        saved_path = os.path.join(args.model_path, args.model_name + '_soft')
    model_name = os.path.join(saved_path, 'net_' + str(epoch + 1) + '.pth')
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1},
               model_name)

def getTrainLoader(args,path):
    read_handle = open(path,'rb')
    dataset = pickle.load(read_handle)
    root = os.path.join(os.getcwd(),'data/market1501/bounding_box_train')
    train_loader = soft_train_loader(args,dataset,root)
    return train_loader
    

def main(args):
    path = os.path.join(os.getcwd(),'soft_label','soft_label_resnet50.txt')
    if not os.path.isfile(path):
        print('soft label file is not exist')


    train_loader = getTrainLoader(args,path)
    _,val_loader, num_query, num_classes, train_size = make_data_loader(args)

    #train_loader, val_loader, num_query, num_classes, train_size = make_data_loader(args)
    model = build_model(args, num_classes)
    optimizer = make_optimizer(args, model)
    scheduler = WarmupMultiStepLR(optimizer, [30, 55], 0.1, 0.01, 5, "linear")

    loss_func = make_loss(args)

    model.to(device)

    for epoch in range(args.Epochs):
        model.train()
        running_loss = 0.0
        running_klloss = 0.0
        running_softloss = 0.0
        running_corrects = 0.0
        for index, data in enumerate(tqdm(train_loader)):
            img, target, soft_target = data
            img = img.cuda()
            target = target.cuda()
            soft_target = soft_target.cuda()
            score,_ = model(img)
            preds = torch.max(score.data, 1)[1]
            loss,klloss,softloss = loss_func(score, target, soft_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_klloss +=klloss.item()
            running_softloss +=softloss.item()
            running_corrects += float(torch.sum(preds == target.data))

        scheduler.step()
        epoch_loss = running_loss / train_size
        epoch_klloss = running_klloss / train_size
        epoch_softloss = running_softloss / train_size
        epoch_acc = running_corrects / train_size
        print("Epoch {}   Loss : {:.4f} KLLoss:{:.8f}  SoftLoss:{:.4f}  Acc:{:.4f}".format(
            epoch, epoch_loss, epoch_klloss,epoch_softloss,epoch_acc))

        if (epoch + 1) % args.n_save == 0:
            evaluator = Evaluator(model, val_loader, num_query)
            cmc, mAP = evaluator.run()
            print('---------------------------')
            print("CMC Curve:")
            for r in [1, 5, 10]:
                print("Rank-{} : {:.1%}".format(r, cmc[r - 1]))
            print("mAP : {:.1%}".format(mAP))
            print('---------------------------')
            save_model(args, model, optimizer, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Konwledge distillation')
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-s', '--image_size', type=list, default=[128, 64])
    parser.add_argument('-p', '--flip_p', type=float, default=0.5)
    parser.add_argument('-pd', '--input_pading', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-m', '--model_name', type=str, default='tiny_resnet50_soft')
    parser.add_argument('-bl', '--base_learning_rate', type=float, default=0.00035)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0005)
    parser.add_argument('-e', '--Epochs', type=int, default=120)
    parser.add_argument('-sampler', '--sampler', type=str, default="soft_label")
    parser.add_argument('-n_save', '--n_save', type=int, default=10)
    parser.add_argument('-mp', '--model_path', type=str, default='./models')
    main(parser.parse_args())
