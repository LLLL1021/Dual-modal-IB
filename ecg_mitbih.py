'''
Training script for ecg classification
'''
from __future__ import print_function

import os
import cv2
import json
import time
import torch
import random
import shutil
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import sklearn.metrics as skm
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import models as models
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ECG LSTM MITBIH Training')
# Datasets
parser.add_argument('-dt', '--dataset', default='ecg', type=str)
#parser.add_argument('-ft', '--transformation', type=str)
parser.add_argument('-ft', '--transformation', default='stft', type=str)
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=300, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=150, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
pp = '/home/ishaan/work_dir/ecg_spectogram/ecg_phase2/checkpoints/cwt_lstm_raw_metadata_w2v_nooverlap/model_best.pth.tar'

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Architecture
parser.add_argument('--depth', type=int, default=110, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: '
                         'Basicblock for ecg)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False,
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu-id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Validate dataset
assert args.dataset == 'ecg', 'Dataset can only be if not args.evaluate:ecg.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

class Ecg_loader(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        super(Ecg_loader, self).__init__()
        print(path)
        self.male_vec = pd.read_csv(os.path.join(path, 'res', 'male.csv'), header=None).to_numpy()[:, 0]
        self.female_vec = pd.read_csv(os.path.join(path, 'res', 'female.csv'), header=None).to_numpy()[:, 0]

        with open(os.path.join(path, 'ecg_labels.json')) as j_file:
            json_data = json.load(j_file)
        self.idx2name = json_data['labels']
        data = json_data['data']
        self.inputs = []
        self.labels = []
        self.gender = []
        self.inputs_full = []
        self.whole_ecg = []
        self.ecg = []
        self.age = []
        for i in tqdm(data):
            subject_img = []
            subject_ecg = []
            a = np.zeros((100))
            for i_name, w_name in zip(i['images'], i['ecg']):
                img = cv2.imread(os.path.join(path, 'images', transform, i_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (90, 90))
                ecg = np.load(os.path.join(path, 'ecg', w_name))
                subject_img.append(np.expand_dims(img.transpose((2, 0, 1)), axis=0))
                subject_ecg.append(np.expand_dims(np.expand_dims(ecg, axis=0), axis=0))
            img_full = cv2.imread(os.path.join(path, 'images_full', transform, i['images_full']))
            img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
            l = i['label']
            a[int(i['age']*100)] = 1
            if i['gender'] == [0, 1]:
                g = self.male_vec
            elif i['gender'] == [1, 0]:
                g = self.female_vec
            self.inputs_full.append(img_full.transpose((2, 0, 1)))
            self.inputs.append(np.concatenate(subject_img, axis=0))
            self.ecg.append(np.concatenate(subject_ecg, axis=0))
            self.whole_ecg.append(np.concatenate(subject_ecg, axis=2))
            self.labels.append(np.array(l))
            self.gender.append(g)
            self.age.append(a)
        print(len(self.whole_ecg))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).float()
        y = torch.from_numpy(np.array(self.labels[idx])).long()
        a = torch.from_numpy(np.array(self.age[idx])).float()
        g = torch.from_numpy(np.array(self.gender[idx])).float()
        w = torch.from_numpy(self.ecg[idx]).float()
        return (x, a, g, w), y

def evaluate(outputs, labels, label_names=None):
    gt = torch.cat(labels, dim=0)
    pred = torch.cat(outputs, dim=0)
    probs = pred
    pred = torch.argmax(pred, dim=1)
    acc = torch.div(100*torch.sum((gt == pred).float()), gt.shape[0])
    name_dict = {0: 'Normal beat (N)', 1: 'Left bundle branch block beat (L)', 2: 'Right bundle branch block beat (R)', 3:
        'Premature ventricular contraction (V)', 4: 'Atrial premature beat (A)', 5: 'Non classified (~)'}

    print('accuracy :', acc)

    gt = gt.cpu().tolist()
    pred = pred.cpu().tolist()

    report = skm.classification_report(
        gt, pred,
        target_names=[name_dict[i] for i in np.unique(gt)],
        digits=3)
    scores = skm.precision_recall_fscore_support(
        gt,
        pred,
        average=None)
    print(report)
    print("F1 Average {:3f}".format(np.mean(scores[2][:3])))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = np.unique(gt).shape[0]
    oh_gt = np.zeros((len(gt), n_classes))
    plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    for i in range(n_classes):
        oh_gt[:, gt == i] = 1
        fpr[i], tpr[i], _ = roc_curve(gt, probs[:, i].cpu(), pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
        lw = 2
        plt.plot(fpr[i], tpr[i], color=colors[i],
                 lw=lw, label=name_dict[i] +' : %0.4f' % roc_auc[i])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Class-Wise AUC and ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.checkpoint, 'roc.png'))
    return 0


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)

    dataloader = Ecg_loader
    train_path = args.data

    traindir = os.path.join(train_path, 'train')
    print(traindir)
    valdir = os.path.join(train_path, 'val')
    if not args.evaluate:
        trainset = dataloader(traindir, transform=args.transformation)
    testset = dataloader(valdir, transform=args.transformation)

    idx2name = testset.idx2name
    label_names = []
    for i in range(0, len(idx2name.keys())):
        label_names.append(idx2name[str(i)])
    num_classes = len(label_names)

    if not args.evaluate:
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model ResNet{}".format(args.depth))

    model = models.__dict__['resnet_lstm_mitbih'](
                num_classes=num_classes,
                depth=args.depth,
                block_name=args.block_name,
            )

    #model = torch.nn.DataParallel(model).cuda()
    model = torch.nn.DataParallel(model).to(device)
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Resume
    title = 'ecg-lstm-resnet' + str(args.depth)
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc= test(testloader, model, criterion, start_epoch, use_cuda, label_names=label_names)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))

        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda, label_names=label_names)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()


    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = (inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda(),
                               inputs[3].cuda()), targets.cuda()
        inputs, targets = (torch.autograd.Variable(inputs[0]), torch.autograd.Variable(inputs[1]),
                           torch.autograd.Variable(inputs[2]),
                           torch.autograd.Variable(inputs[3])), torch.autograd.Variable(targets)

        outputs = model(inputs)
        print(targets)
        print("----------")
        print(outputs)
        targets= torch.argmax(targets ,axis=1)
        #targets=targets.squeeze(1)
        print("==========")
        print(targets)
        loss = criterion(outputs, targets)
        print(loss)
        

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 6))

        if float(torch.__version__[:3]) < 0.5:
            losses.update(loss.data[0], inputs[0].size(0))
            top1.update(prec1[0], inputs[0].size(0))
            top5.update(prec5[0], inputs[0].size(0))
        else:
            losses.update(loss.data, inputs[0].size(0))
            top1.update(prec1, inputs[0].size(0))
            top5.update(prec5, inputs[0].size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    # evaluate(pred, gt)
    return (losses.avg, top1.avg)


def test(testloader, model, criterion, epoch, use_cuda, label_names=None):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    gt = []
    pred = []
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = (inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda(),
                               inputs[3].cuda()), targets.cuda()
        inputs, targets = (torch.autograd.Variable(inputs[0]), torch.autograd.Variable(inputs[1]),
                           torch.autograd.Variable(inputs[2]),
                           torch.autograd.Variable(inputs[3])), torch.autograd.Variable(targets)

        # compute output
        st = time.time()
        outputs = model(inputs)
        # print(time.time()-st)
        targets= torch.argmax(targets ,axis=1)
        #targets=targets.squeeze(1)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        gt.append(targets.data)
        pred.append(outputs.data)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 6))
        if float(torch.__version__[:3]) < 0.5:
            losses.update(loss.data[0], inputs[0].size(0))
            top1.update(prec1[0], inputs[0].size(0))
            top5.update(prec5[0], inputs[0].size(0))
        else:
            losses.update(loss.data, inputs[0].size(0))
            top1.update(prec1, inputs[0].size(0))
            top5.update(prec5, inputs[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    evaluate(pred, gt, label_names=label_names)
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
