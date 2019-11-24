import argparse
import os
import sys
import torch
import torchvision
from tqdm import tqdm
import time
from Logger import Logger

from data import make_data_loader
from model import build_model
from solver import make_optimizer,WarmupMultiStepLR
from layer import make_loss
from evaluate import Evaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet50(pretrained=True)

def save_model(args,model,optimizer,epoch):
    saved_path = os.path.join(args.model_path,args.model_name+args.modify)
    model_name = os.path.join(saved_path,'net_'+str(epoch + 1)+'.pth')
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1},
               model_name)


def main(args):
    sys.stdout = Logger(
        os.path.join(args.log_path, args.log_description, 'log' + time.strftime(".%m_%d_%H:%M:%S") + '.txt'))

    train_loader,val_loader,num_query,num_classes,train_size = make_data_loader(args)
    model = build_model(args,num_classes)
    print(model)
    optimizer = make_optimizer(args,model)
    scheduler = WarmupMultiStepLR(optimizer,[30,55],0.1,0.01,5,"linear")

    loss_func = make_loss(args)

    model.to(device)

    for epoch in range(args.Epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        for index,data in enumerate(tqdm(train_loader)):
            img,target = data
            img = img.cuda()
            target = target.cuda()
            score,_ = model(img)
            preds = torch.max(score.data,1)[1]
            loss = loss_func(score,target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects +=float(torch.sum(preds == target.data))
            
        scheduler.step()
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects / train_size
        print("Epoch {}   Loss : {:.6f}   Acc:{:.4f}".format(epoch, epoch_loss, epoch_acc))

        if (epoch+1)%args.n_save==0:
            evaluator = Evaluator(model,val_loader,num_query)
            cmc,mAP = evaluator.run()
            print('---------------------------')
            print("CMC Curve:")
            for r in [1,5,10]:
                print("Rank-{} : {:.1%}".format(r,cmc[r-1]))
            print("mAP : {:.1%}".format(mAP))
            print('---------------------------')
            save_model(args,model,optimizer,epoch)

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Konwledge distillation')
    parser.add_argument('-d','--dataset',type=str,default='market1501')
    parser.add_argument('-s','--image_size',type=list,default=[128,64])
    parser.add_argument('-p','--flip_p',type=float,default=0.5)
    parser.add_argument('-pd','--input_pading',type=int,default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-m', '--model_name', type=str, default='resnet50')
    parser.add_argument('-mo', '--modify', type=str, default='_2048_partial')
    parser.add_argument('-bl', '--base_learning_rate', type=float, default=0.00035)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0005)
    parser.add_argument('-e', '--Epochs', type=int, default=120)
    parser.add_argument('-sampler', '--sampler', type=str, default="softmax")
    parser.add_argument('-n_save', '--n_save', type=int, default=10)
    parser.add_argument('-mp', '--model_path', type=str, default='./models')

    parser.add_argument('-lp', '--log_path', type=str, default='./logs')
    parser.add_argument('-ld', '--log_description', type=str, default='tiny_resnet50_partial')

    parser.add_argument('-mr', '--margin_tri', type=float, default=0.02)
    main(parser.parse_args())




