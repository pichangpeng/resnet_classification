import argparse
import os
import shutil
import time
from torchvision.transforms.transforms import Pad
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import resnet
from  dataset import ImageDataset
from sklearn import metrics
from sklearn.metrics import auc
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))
print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=24, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--threshold',default=0.3, type=float,help="the threshold of computing the precision and recall")
parser.add_argument('--num-cls',default=4, type=int,help="the num of class")
best_prec1 = 0
args = parser.parse_args()

def main():
    global args, best_prec1


    # Check the save_dir exists or not
    if not os.path.exists(args.arch):
        os.makedirs(args.arch)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transforms_ = [ transforms.Pad(40),
                    transforms.Resize(400),
                    transforms.CenterCrop((320,320)), 
                    transforms.ToTensor(),
                    normalize]

    train_loader = torch.utils.data.DataLoader(
        ImageDataset("../classification/trainSet","../classification/train.json",transforms_),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ImageDataset("../classification/testSet","../classification/test.json",transforms_),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(torch.tensor([1,0.2,0.3,0.1])).cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[int(args.epochs*0.4), int(args.epochs*0.8)], last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion,epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.arch, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.arch, 'model.th'))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    # for i, (input, target) in enumerate(train_loader):
    for i, batch in enumerate(train_loader):
        input=batch["images"]
        target=batch["labels"]
        # print(input)
        # print(target)
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion,epoch=0):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    output_=torch.Tensor()
    target_=torch.Tensor()
    image_name_=[]
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input=batch["images"]
            target=batch["labels"]
            image_name=batch["image_names"]
            target = target.cuda()
            input_var = input.cuda()
            target_var = target

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            
            output = output.float()
            loss = loss.float()
            
            output_=torch.cat((output_,output.cpu()),0)
            target_=torch.cat((target_,target.cpu()),0)
            image_name_.extend(image_name)
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))
    precision,recall,hard_negative=precision_recall(output_.data,target_,image_name_)
    roc(output_.data,target_,epoch)

    if not os.path.exists("%s/hard_negative/%d"%(args.arch,epoch)):
        os.makedirs("%s/hard_negative/%d"%(args.arch,epoch))
    for i,name_info in tqdm(enumerate(hard_negative)):
        name=name_info[0]
        info=name_info[1]
        shutil.copyfile("../classification/testSet/%s"%name,"./%s/hard_negative/%d/%d_%s"%(args.arch,epoch+1,i+1,info+"."+name.split(".")[-1]))
    print("Precision:{}\t" "Recall:{}\t" "hard_negative num:{}\t".format(precision,recall,len(hard_negative)))
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def precision_recall(output,target,image_name_,topk=(1,)):
    precision=[]
    recall=[]
    hard_negitive=[]
    correct=[0]*args.num_cls
    maxk = max(topk)
    batch_size=target.size(0)
    softmax_func=nn.Softmax(dim=1)
    pre=softmax_func(output)
    prob, pred = pre.topk(maxk, 1, True, True)
    target=target.numpy().tolist()
    prob=prob.view(-1).numpy().tolist()
    pred=pred.view(-1).numpy().tolist()
    for i in range(batch_size):
        if prob[i]>args.threshold:
            if target[i]==pred[i]:
                correct[int(target[i])]=correct[int(target[i])]+1
            else:
                hard_negitive.append([image_name_[i],"%d_%d"%(target[i],pred[i])])
        else:
            hard_negitive.append([image_name_[i],"%d_%d"%(target[i],pred[i])])
    for i in range(args.num_cls):
        pred_count=pred.count(i)
        target_count=target.count(i)
        if pred_count==0:
            precision.append(-1)
        else:
            precision.append(correct[i]/pred.count(i))
        if target_count==0:
            recall.append(-1)
        else:
            recall.append(correct[i]/target.count(i))
    return precision,recall,hard_negitive

def roc(output,target,epoch,topk=(args.num_cls,)):
    if not os.path.exists("%s/roc"%args.arch):
        os.makedirs("%s/roc"%args.arch)
    score=[]
    maxk = max(topk)
    batch_size=target.size(0)
    target=target.numpy().tolist()
    softmax_func=nn.Softmax(dim=1)
    pre=softmax_func(output).numpy()
    for i in range(batch_size):
        score.append(pre[i][int(target[i])])
    plt.figure()
    for cls in range(maxk):
        fpr, tpr, _ = metrics.roc_curve(target, score, pos_label=cls)
        plt.plot(fpr, tpr, label="%d"%cls,color=["r","b","y","g","c"][cls])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig("%s/roc/roc_%d.png"%(args.arch,epoch+1))
    plt.show()

if __name__ == '__main__':
    main()
    # pre=torch.rand(10,5)
    # target=torch.tensor([0,1,3,2,4,0,4,2,1,3])
    # a=torch.Tensor()
    # print(torch.cat((target,target),0))
    # print(pre)
    # print(target)
    # softmax_func=nn.Softmax(dim=1)
    # pre=softmax_func(pre)
    # print(pre)
    # print(pre[:,target.numpy().tolist()])
    # _, pred = pre.topk(5, 1, True, True)
    # print(_)
    # print(pred)
    # print(target.numpy().tolist().count(1))
    # precision,recall=precision_recall(pre,target)
    # print(precision)
    # print(recall)
    # roc(pre,target)