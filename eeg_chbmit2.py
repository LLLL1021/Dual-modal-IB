'''
Training script for ecg classification
'''
from __future__ import print_function
import tensorflow as tf
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

import warnings
warnings.filterwarnings('ignore')
#print("11111")
#import tensorflow as tf
#print("22222")
#import sys

from sklearn.model_selection import train_test_split

#import import_data as impt
from helper import f_get_minibatch_set, evaluate1

from class_DeepIMV_AISTATS import DeepIMV_AISTATS


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ECG LSTM MITBIH Training')


#def init_arg():

# Datasets
parser.add_argument('-dt', '--dataset', default='ecg', type=str)
# parser.add_argument('-ft', '--transformation', type=str)
parser.add_argument('-ft', '--transformation', default='stft', type=str)
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=30, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
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
parser.add_argument('--depth', type=int, default=50, help='Model depth.')
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
parser.add_argument('--gpu-id', default='2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Validate dataset
assert args.dataset == 'ecg', 'Dataset can only be if not args.evaluate:ecg.'

# def init_arg():
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1234, help='random seed', type=int)

parser.add_argument('--h_dim_p', default=20, help='number of hidden nodes -- predictor', type=int)
parser.add_argument('--num_layers_p', default=2, help='number of layers -- predictor', type=int)

parser.add_argument('--h_dim_e', default=20, help='number of hidden nodes -- encoder', type=int)
parser.add_argument('--num_layers_e', default=3, help='number of layers -- encoder', type=int)

parser.add_argument('--z_dim', default=5, help='dimension of latent representations', type=int)

parser.add_argument("--lr_rate", default=0.00001, help='learning rate', type=float)
parser.add_argument("--l1_reg", default=0., help='l1-regularization', type=float)

parser.add_argument("--itrs", default=1, type=int)
parser.add_argument("--step_size", default=1, type=int)
parser.add_argument("--max_flag", default=0.2, type=int)

parser.add_argument("--mb_size", default=1, type=int)
parser.add_argument("--keep_prob", help='keep probability for dropout', default=0.7, type=float)

parser.add_argument('--alpha', default=1.0, help='coefficient -- alpha', type=float)
parser.add_argument('--beta', default=0.01, help='coefficient -- beta', type=float)

parser.add_argument('--save_path', default='./storage/', help='path to save files', type=str)

#return parser.parse_args()


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
use_cuda = torch.cuda.is_available()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
        #print('json_data :',json_data)
        #print('idx2name :', self.idx2name)
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
           # print('l :', l)
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
       # print(len(self.whole_ecg))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).float()
        y = torch.from_numpy(np.array(self.labels[idx])).long()
        a = torch.from_numpy(np.array(self.age[idx])).float()
        g = torch.from_numpy(np.array(self.gender[idx])).float()
        w = torch.from_numpy(self.ecg[idx]).float()
        #print('x :', x)
        #print('y :', y)
       # print('a :', a)
       # print('g :', g)
        #print('w :', w)
        return (x, a, g, w), y

def evaluate(outputs, labels, label_names=None):
    gt = torch.cat(labels, dim=0)
   # print("evaluate.gt : ", gt)
    pred = torch.cat(outputs, dim=0)
    probs = pred
   # print("evaluate.pred1 : ",pred)
    pred = torch.argmax(pred, dim=1)
  #  print("evaluate.pred2 : ", pred)
    acc = torch.div(100*torch.sum((gt == pred).float()), gt.shape[0])
    #name_dict = {0: 'Normal beat (N)', 1: 'Left bundle branch block beat (L)', 2: 'Right bundle branch block beat (R)', 3:
    #    'Premature ventricular contraction (V)', 4: 'Atrial premature beat (A)', 5: 'Non classified (~)'}
    name_dict = {0: 'No dianxian (N)', 1: 'dianxian (Y)',  5: 'Non classified (~)'}
    print('accuracy :', acc)

    gt = gt.cpu().tolist()
    pred = pred.cpu().tolist()

    #print('gt :', len(gt))
    #print('pred :', len(pred))
    
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

    #acc_train, sensitivity, specificity, F1 = accuracy1(pred, gt)
    #print(" acc_train, sensitivity, specificity, F1 : ", acc_train, sensitivity, specificity, F1)

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
    plt.close()
    return 0


def accuracy1(output, labels):
    # output = output[0]
    # print("output",output)
    #print("output.shape,labels.shape",output.shape,labels.shape)
    #print("output",output,"output.max(1)",output.max(1),"output.max(1)[0]",output.max(1)[0])
    #preds = torch.round(output.max(1)[0]).type_as(labels) #A.max(1)  # 返回A每一行最大值组成的一维数组和这个数的索引组成的

    #preds = output.max(1)[1].type_as(labels)
    preds=output
    # print('output.max(1)[0]',output.max(1)[0])
    # save_excel(output.max(1)[0].cpu().numpy(), labels.cpu().numpy())
    # preds = torch.where(output.cpu() > 0.5, torch.tensor(1), torch.tensor(0)).type_as(labels)
    # preds = torch.squeeze(preds)
    # print(preds)
    # print("preds:",preds)
    # print("labels:",labels)
    TP = TN = FN = FP = 0
    # TP    predict 和 label 同时为1
    TP += ((preds == 1) & (labels.data == 1)).sum()
    # TN    predict 和 label 同时为0
    TN += ((preds == 0) & (labels.data == 0)).sum()
    # FN    predict 0 label 1
    FN += ((preds == 0) & (labels.data == 1)).sum()
    # FP    predict 1 label 0
    FP += ((preds == 1) & (labels.data == 0)).sum()
    # print("TP",TP,"TN",TN,"FN",FN,"FP",FP)
    sensitivity = TP * 1.0 / (TP + FP)
    specificity = TP * 1.0 / (TP + FN)
    # print("sensitivity:",sensitivity,"specificity:",specificity)

    if (TP + FN) == 0: #说明没有label=1的，即这个batch不含癫痫发作的片段
        pass
    else:
        L1 = TP * 1.0 / (TP + FN)
        L2 = TN * 1.0 / (TN + FP)
        print("label=1,num:",(labels.data == 1).sum().data.item(),"label=0,num:",(labels.data == 0).sum().data.item(),"seizure acc:", L1.data.item(), "no-seizure acc:", L2.data.item())
    F1 = 2 * specificity * sensitivity / (specificity + sensitivity)
    # print("F1:", F1)
    acc = (TP + TN) * 1.0 / (TP + TN + FP + FN)
    return acc,sensitivity,specificity,F1


def main():
    #print("11111")
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    #print("22222")
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
   # print("33333333")
    # Data
    print('==> Preparing dataset %s' % args.dataset)

    dataloader = Ecg_loader
    train_path = args.data

    traindir = os.path.join(train_path, 'train')
    #print(traindir)
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
    #xs,wecg=model
    #print("xs,wecg",xs,wecg)
    #print(model)
    model = torch.nn.DataParallel(model).cuda()
    #model = torch.nn.DataParallel(model).to(device)
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MultiLabelSoftMarginLoss()
    #criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Resume
    title = 'ecg-lstm-resnet' + str(args.depth)
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        print(os.path.dirname(args.resume))
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
        #test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda, label_names=label_names)
        #train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda, label_names=label_names)
        #print("test_loss : ",test_loss)
        #print("test_acc : ", test_acc)
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
    # switch to train model
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
        # (x, a, g, w), y
        # print("inputs : ",input)
        if use_cuda:
            inputs, targets = (inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda(),
                               inputs[3].cuda()), targets.cuda()
        inputs, targets = (torch.autograd.Variable(inputs[0]), torch.autograd.Variable(inputs[1]),
                           torch.autograd.Variable(inputs[2]),
                           torch.autograd.Variable(inputs[3])), torch.autograd.Variable(targets)
        # print("inputs21 : ", inputs[0])
        # print("inputs2 : ", inputs[0].cuda())
        # outputs,w = model(inputs)
        #x1, x2 = model(inputs)
        #print("=============")
        #X_set,ou = model(inputs).tolist()
        #print("inputs", inputs)
        X_set, ou , outputs = model(inputs) #eeg 光谱图
       # print("X_set0", X_set)
       # print("ou", ou)
       # print("targets", targets)
        targets1 = targets[:, :1].reshape(-1).long()
        #print("targets2", targets1)
        loss = criterion(outputs, targets1)

        prec1, prec5 = accuracy(outputs.data, targets1.data, topk=(1, 2))

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
        #print(loss)
        #print("-------------------TRAIN_LOSS", loss)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        X_set=X_set.tolist()
       # print("------------")
        #print(X_set.size())
        #X_set = model(inputs)
        X_set=np.array(X_set)
        #print(X_set.shape())
        Y_onehot=targets
        #Y_onehot=torch.tensor(Y_onehot)
        Y_onehot=np.array(Y_onehot)
        M = len(X_set)
        ### Construct Mask Vector to indicate available (m=1) or missing (m=0) values
        Mask = np.ones([np.shape(X_set[0])[0], M])
        for m_idx in range(M):
            Mask[np.isnan(X_set[m_idx]).all(axis=1), m_idx] = 0
            X_set[m_idx][Mask[:, m_idx] == 0] = np.mean(X_set[m_idx][Mask[:, m_idx] == 1], axis=0)

        #args = init_arg()
        #seed = args.seed
        seed=1234
        ### import multi-view dataset with arbitrary view-missing patterns.
        #X_set, Y_onehot, Mask = impt.import_incomplete_handwritten()
        #X_set=x1
        #print("X_set", X_set)
        tr_X_set, te_X_set, va_X_set = {}, {}, {}
        for m in range(len(X_set)):
            #print("=====")
            #print("m1",m)
            #print(" len(X_set) : ", len(X_set))
            tr_X_set[m], te_X_set[m] = train_test_split(X_set[m], test_size=0.2, random_state=seed)
            tr_X_set[m], va_X_set[m] = train_test_split(tr_X_set[m], test_size=0.2, random_state=seed)

       # print("tr_X_set",tr_X_set)
        #print("te_X_set", te_X_set)
        #print("va_X_set", va_X_set)

        #tr_X_set, va_X_set = train_test_split(X_set, test_size=0.2, random_state=seed)
        #print("Y_onehot",Y_onehot)
        tr_Y_onehot, te_Y_onehot, tr_M, te_M = train_test_split(Y_onehot, Mask, test_size=0.2, random_state=seed)
        #print("tr_Y_onehot", tr_Y_onehot)
        tr_Y_onehot, va_Y_onehot, tr_M, va_M = train_test_split(tr_Y_onehot, tr_M, test_size=0.2, random_state=seed)

        #print("tr_Y_onehot2", tr_Y_onehot)
       # print("te_Y_onehot", te_Y_onehot)
        #print("tr_M", tr_M)
        #print("te_M", te_M)
        #print("va_M", va_M)
        #print(" m2: ", m)
        #x_dim_set = [te_X_set[m].shape[1] for m in range(len(te_X_set))]
        x_dim_set = [tr_X_set[m].shape[1] for m in range(len(tr_X_set))]
        #print(" m: ", m)
        #print(" te_X_set[m].shape[1] : ", te_X_set[m].shape[1])
        #print(" len(te_X_set) : ", len(te_X_set))
        #x_dim_set = [tr_X_set.shape[1]]
        y_dim = np.shape(tr_Y_onehot)[1]
        #print(" te_Y_onehot : ", tr_Y_onehot)
        #print(" x_dim_set : ", x_dim_set)
        #print(" y_dim : ", y_dim)
        #if y_dim == 1:
        #    y_type = 'continuous'
        #elif y_dim == 2:
        #    y_type = 'binary'
        #else:
        #    y_type = 'categorical'
        y_type = 'categorical'
        #print(" y_type : ", y_type)
        #mb_size = args.mb_size
        mb_size=2
        z_dim=5
        steps_per_batch = int(np.shape(tr_M)[0] / mb_size)  # for moving average

        input_dims = {
            'x_dim_set': x_dim_set,
            'y_dim': y_dim,
            'y_type': y_type,
            #'z_dim': args.z_dim,
            'z_dim': z_dim,

            'steps_per_batch': steps_per_batch
        }

        h_dim_p=20
        h_dim_e=20
        num_layers_p=2
        num_layers_e=3
        l1_reg: float = 0.
        lr_rate: float = 0.000000001
        itrs=1
        step_size=1
        max_flag=0.2

        keep_prob: float = 0.7

        alpha: float = 1.0
        beta: float = 0.01
        save_path='/icislab/volume1/wjl/eeggai/check/'
        network_settings = {
            # 'h_dim_p1': args.h_dim_p,
            # 'num_layers_p1': args.num_layers_p,  # view-specific
            #
            # 'h_dim_p2': args.h_dim_p,
            # 'num_layers_p2': args.num_layers_p,  # multi-view
            #
            # 'h_dim_e': args.h_dim_e,
            # 'num_layers_e': args.num_layers_e,
            #
            # 'fc_activate_fn': tf.nn.relu,
            # 'reg_scale': args.l1_reg,

            'h_dim_p1': h_dim_p,
            'num_layers_p1': num_layers_p,  # view-specific

            'h_dim_p2': h_dim_p,
            'num_layers_p2': num_layers_p,  # multi-view

            'h_dim_e': h_dim_e,
            'num_layers_e': num_layers_e,

            'fc_activate_fn': tf.nn.relu,
            'reg_scale': l1_reg,
        }

        # lr_rate = args.lr_rate
        # iteration = args.itrs
        # stepsize = args.step_size
        # max_flag = args.max_flag
        #
        # k_prob = args.keep_prob
        #
        # alpha = args.alpha
        # beta = args.beta
        #
        # save_path = args.save_path
        lr_rate = lr_rate
        iteration = 50
        stepsize = step_size
        max_flag = max_flag

        k_prob = keep_prob

        alpha = alpha
        beta = beta

        save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        tf.reset_default_graph()
        gpu_options = tf.GPUOptions()

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #print("000000000000000000000000000000000")
        model1 = DeepIMV_AISTATS(sess, "DeepIMV_AISTATS", input_dims, network_settings)  

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        ##### TRAINING
        min_loss = 1e+8
        max_acc = 0.0

        tr_avg_Lt, tr_avg_Lp, tr_avg_Lkl, tr_avg_Lps, tr_avg_Lkls, tr_avg_Lc = 0, 0, 0, 0, 0, 0
        #tr_avg_Lt, tr_avg_Lp, tr_avg_Lkl, tr_avg_Lps, tr_avg_Lkls, tr_avg_Lc = 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8
        va_avg_Lt, va_avg_Lp, va_avg_Lkl, va_avg_Lps, va_avg_Lkls, va_avg_Lc = 0, 0, 0, 0, 0, 0
        stop_flag = 0
        for itr in range(iteration):
            #print("itr : ",itr)
            x_mb_set, y_mb, m_mb = f_get_minibatch_set(mb_size, tr_X_set, tr_Y_onehot, tr_M)

            _, Lt, Lp, Lkl, Lps, Lkls, Lc = model1.train(x_mb_set, y_mb, m_mb, alpha, beta, lr_rate, k_prob)

            tr_avg_Lt += Lt / stepsize
            tr_avg_Lp += Lp / stepsize
            tr_avg_Lkl += Lkl / stepsize
            tr_avg_Lps += Lps / stepsize
            tr_avg_Lkls += Lkls / stepsize
            tr_avg_Lc += Lc / stepsize

            x_mb_set, y_mb, m_mb = f_get_minibatch_set(min(np.shape(va_M)[0], mb_size), va_X_set, va_Y_onehot, va_M)
            Lt, Lp, Lkl, Lps, Lkls, Lc, _, _ = model1.get_loss(x_mb_set, y_mb, m_mb, alpha, beta)

            va_avg_Lt += Lt / stepsize
            va_avg_Lp += Lp / stepsize
            va_avg_Lkl += Lkl / stepsize
            va_avg_Lps += Lps / stepsize
            va_avg_Lkls += Lkls / stepsize
            va_avg_Lc += Lc / stepsize

            if (itr + 1) % stepsize == 0:
                y_pred, y_preds = model1.predict_ys(va_X_set, va_M)

                #         score =

                #print(
                #    "{:05d}: TRAIN| Lt={:.3f} Lp={:.3f} Lkl={:.3f} Lps={:.3f} Lkls={:.3f} Lc={:.3f} | VALID| Lt={:.3f} Lp={:.3f} Lkl={:.3f} Lps={:.3f} Lkls={:.3f} Lc={:.3f} score={}".format(
                #        itr + 1, tr_avg_Lt, tr_avg_Lp, tr_avg_Lkl, tr_avg_Lps, tr_avg_Lkls, tr_avg_Lc,
                #        va_avg_Lt, va_avg_Lp, va_avg_Lkl, va_avg_Lps, va_avg_Lkls, va_avg_Lc,
                 #       evaluate1(va_Y_onehot, np.mean(y_preds, axis=0), y_type))
                #)
                #print("======Vva_Y_onehot : ", va_Y_onehot)
                #print("======Vy_preds : ", y_preds)
               # print("======Vpred_y : ", np.mean(y_preds, axis=0))
                if min_loss > va_avg_Lt:
                    min_loss = va_avg_Lt
                    stop_flag = 0
                    saver.save(sess, save_path + 'best_model')
                    #print('saved...')
                else:
                    stop_flag += 1

                tr_avg_Lt, tr_avg_Lp, tr_avg_Lkl, tr_avg_Lps, tr_avg_Lkls, tr_avg_Lc = 0, 0, 0, 0, 0, 0
                va_avg_Lt, va_avg_Lp, va_avg_Lkl, va_avg_Lps, va_avg_Lkls, va_avg_Lc = 0, 0, 0, 0, 0, 0

                if stop_flag >= max_flag:
                    break

        #print('FINISHED...')
        saver.restore(sess, save_path + 'best_model')

        _, pred_ys = model1.predict_ys(te_X_set, te_M)
        pred_y = np.mean(pred_ys, axis=0)
        #print("======Tte_Y_onehot : ",te_Y_onehot)
        #print("======Tpred_ys : ", pred_ys)
        #print("======Tpred_y : ", pred_y)
        #print('Test Score: {}'.format(evaluate1(te_Y_onehot, pred_y, y_type)))



        #stop_flag = 0
        #x_mb_set, y_mb, m_mb = f_get_minibatch_set(mb_size, te_X_set, te_Y_onehot, te_M)
        #_, Lt, Lp, Lkl, Lps, Lkls, Lc = model1.train(x_mb_set, y_mb, m_mb, alpha, beta, lr_rate, k_prob)


        #tr_avg_Lt += Lt / stepsize
        #tr_avg_Lp += Lp / stepsize
        #tr_avg_Lkl += Lkl / stepsize
        #tr_avg_Lps += Lps / stepsize
        #tr_avg_Lkls += Lkls / stepsize
        #tr_avg_Lc += Lc / stepsize

        #itr = 0
        #print(
        #    "{:05d}: TRAIN| Lt={:.3f} Lp={:.3f} Lkl={:.3f} Lps={:.3f} Lkls={:.3f} Lc={:.3f} ".format(
        #        itr + 1, tr_avg_Lt, tr_avg_Lp, tr_avg_Lkl, tr_avg_Lps, tr_avg_Lkls, tr_avg_Lc)
       # )
        #stop_flag += 1

        #tr_avg_Lt, tr_avg_Lp, tr_avg_Lkl, tr_avg_Lps, tr_avg_Lkls, tr_avg_Lc = 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8
        #va_avg_Lt, va_avg_Lp, va_avg_Lkl, va_avg_Lps, va_avg_Lkls, va_avg_Lc = 0, 0, 0, 0, 0, 0


    #print('FINISHED...')

    return (losses.avg, top1.avg)

def train1(trainloader, model, criterion, optimizer, epoch, use_cuda):
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
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 2))

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
        #print("TRAIN_LOSS",loss)

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
        X,k,outputs = model(inputs)
        targets=targets[: , :1].reshape(-1).long()
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        gt.append(targets.data)
        pred.append(outputs.data)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 2))
        if float(torch.__version__[:3]) < 0.5:
            losses.update(loss.data[0], inputs[0].size(0))
            top1.update(prec1[0], inputs[0].size(0))
            top5.update(prec5[0], inputs[0].size(0))
        else:
            losses.update(loss.data, inputs[0].size(0))
            top1.update(prec1, inputs[0].size(0))
            top5.update(prec5, inputs[0].size(0))
        # measure elapsed time
        #print("-------------------TEST_LOSS",loss)
        batch_time.update(time.time() - end)
        end = time.time()
    evaluate(pred, gt, label_names=label_names)
    return (losses.avg, top1.avg)


def test1(testloader, model, criterion, epoch, use_cuda, label_names=None):
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
        X,outputs = model(inputs)
        #print("targetstest : ",targets)
        #print("outputs : ",outputs)

        # print(time.time()-st)
        #targets= torch.argmax(targets ,axis=1)
        #targets=targets.squeeze(1)
        targets=targets[: , :1].reshape(-1).long()
       # print("outputs2 :",outputs)
       # print("targets", targets)
        loss = criterion(outputs, targets)
        #print("targetstest2 : ",targets)
        # measure accuracy and record loss
        gt.append(targets.data)
        pred.append(outputs.data)
        #print("targets.data : ",targets.data)
        #print("outputs.data : ",outputs.data)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 2))
        if float(torch.__version__[:3]) < 0.5:
            losses.update(loss.data[0], inputs[0].size(0))
            top1.update(prec1[0], inputs[0].size(0))
            top5.update(prec5[0], inputs[0].size(0))
        else:
            losses.update(loss.data, inputs[0].size(0))
            top1.update(prec1, inputs[0].size(0))
            top5.update(prec5, inputs[0].size(0))

        # measure elapsed time
        print("TEST_LOSS",loss)
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
