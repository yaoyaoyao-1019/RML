# train a cheap teacher with an adversarial network

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, csgraph

from tqdm import tqdm
import argparse
import os
import random
import logging
import numpy as np

from utils.utils import RunningAverage, set_logger, Params
from model import *
from data_loader import fetch_dataloader

# ************************** random seed **************************
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str,
                    default='/home/yao/code/Nasty-Teacher-main/cheap_exp/TinyImage/kd_cheap_resnet50/cheap_resnet50')
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

device_ids = args.gpu_id
torch.cuda.set_device(device_ids[0])


# ************************** training function **************************
def train_epoch_kd_adv(model, model_ad, optim, data_loader, epoch, params):
    model.train()
    model_ad.train()
    for p in model_ad.parameters():
        p.requires_grad = False
    tch_loss_avg = RunningAverage()
    ad_loss_avg = RunningAverage()
    loss_avg = RunningAverage()

    with tqdm(total=len(data_loader)) as t:  # Use tqdm for progress bar
        for i, (train_batch, labels_batch) in enumerate(data_loader):
            if params.cuda:
                train_batch = train_batch.cuda()  # (B,3,32,32)
                labels_batch = labels_batch.cuda()  # (B,)

            # compute (teacher) model output and loss
            output_tch = model(train_batch)  # logit without SoftMax

            # teacher loss: CE(output_tch, label)
            tch_loss = nn.CrossEntropyLoss()(output_tch, labels_batch)

            # ############ adversarial loss ####################################
            # computer adversarial model output
            # with torch.no_grad():
            #     output_stu = model_ad(train_batch)  # logit without SoftMax
            output_stu = model_ad(train_batch)  # logit without SoftMax
            output_stu = output_stu.detach()

            # adversarial loss: KLdiv(output_stu, output_tch)
            T = params.temperature
            alpha = params.alpha
            beta = params.beta
            # adv_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output_stu / T, dim=1),
            #                           F.softmax(output_tch / T, dim=1)) * (T * T)   # wish to max this item
            anti_loss = cheap_loss(output_tch, output_stu, labels_batch, alpha, beta, T)

            # total loss
            loss = tch_loss + anti_loss

            # ############################################################

            optim.zero_grad()
            loss.backward()
            optim.step()

            # update the average loss
            loss_avg.update(loss.item())
            tch_loss_avg.update(tch_loss.item())
            ad_loss_avg.update(anti_loss.item())

            # tqdm setting
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    return loss_avg(), tch_loss_avg(), ad_loss_avg()


def evaluate(model, loss_fn, data_loader, params):
    model.eval()
    # summary for current eval loop
    summ = []

    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in data_loader:
            if params.cuda:
                data_batch = data_batch.cuda()          # (B,3,32,32)
                labels_batch = labels_batch.cuda()      # (B,)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            # calculate accuracy
            output_batch = np.argmax(output_batch, axis=1)
            acc = 100.0 * np.sum(output_batch == labels_batch) / float(labels_batch.shape[0])

            summary_batch = {'acc': acc, 'loss': loss.item()}
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    return metrics_mean


def train_and_eval_kd_adv(model, model_ad, optim, train_loader, dev_loader, params):
    best_val_acc = -1
    best_epo = -1
    lr = params.learning_rate

    for epoch in range(params.num_epochs):
        lr = adjust_learning_rate(optim, epoch, lr, params)
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        logging.info('Learning Rate {}'.format(lr))

        # ********************* one full pass over the training set *********************
        train_loss, train_tloss, train_aloss = train_epoch_kd_adv(model, model_ad, optim,
                                                                  train_loader, epoch, params)
        logging.info("- Train loss : {:05.3f}".format(train_loss))
        logging.info("- Train teacher loss : {:05.3f}".format(train_tloss))
        logging.info("- Train adversarial loss : {:05.3f}".format(train_aloss))

        # ********************* Evaluate for one epoch on validation set *********************
        val_metrics = evaluate(model, nn.CrossEntropyLoss(), dev_loader, params)  # {'acc':acc, 'loss':loss}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
        logging.info("- Eval metrics : " + metrics_string)

        # save model
        save_name = os.path.join(args.save_path, 'last_model.tar')
        torch.save({
            'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim.state_dict()},
            save_name)

        # ********************* get the best validation accuracy *********************
        val_acc = val_metrics['acc']
        if val_acc >= best_val_acc:
            best_epo = epoch + 1
            best_val_acc = val_acc
            logging.info('- New best model ')
            # save best model
            save_name = os.path.join(args.save_path, 'best_model.tar')
            torch.save({
                'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim.state_dict()},
                save_name)

        logging.info('- So far best epoch: {}, best acc: {:05.3f}'.format(best_epo, best_val_acc))


def adjust_learning_rate(opt, epoch, lr, params):
    if epoch in params.schedule:
        lr = lr * params.gamma
        for param_group in opt.param_groups:
            param_group['lr'] = lr
    return lr


def swap_original(original, cheap):
    # rearrange original logits
    swap = torch.zeros_like(original, device=original.device)
    dis = cheap[:, :, np.newaxis] - original[:, np.newaxis, :]
    dis = dis.cpu()

    for i in range(dis.shape[0]):
        # row, col = linear_sum_assignment(dis[i].cpu().detach().numpy(), True)
        # swap[i] = original[i, col]
        # 将 dis 张量转换为一个 CSR 矩阵
        dis_csr = csr_matrix(dis[i].cpu().detach().numpy())
        # 计算最小权重匹配
        row, col = csgraph.min_weight_full_bipartite_matching(dis_csr, True)
        # 将结果复制回 GPU
        row = torch.from_numpy(row.astype(np.int64)).cuda()
        col = torch.from_numpy(col.astype(np.int64)).cuda()
        swap[i] = original[i, col]
    return swap


def rank_reverse(logits):
    _, indices = torch.sort(logits, dim=1)
    n = int(indices.shape[1] / 2)

    for i in range(1, n):
        indices[:, [i-1, -i-1]] = indices[:, [-i-1, i-1]]

    rows = torch.arange(indices.shape[0]).unsqueeze(1)
    reversed_logits = logits[rows, indices]
    return reversed_logits


def cheap_loss(logits_cheap, logits_origin, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_cheap, target)
    other_mask = _get_other_mask(logits_cheap, target)
    pred_chep = F.softmax(logits_cheap / temperature, dim=1)
    pred_origin = F.softmax(logits_origin / temperature, dim=1)
    pred_chep = cat_mask(pred_chep, gt_mask, other_mask)
    pred_origin = cat_mask(pred_origin, gt_mask, other_mask)
    log_pred_student = torch.log(pred_chep)
    positive_loss = F.kl_div(log_pred_student, pred_origin, reduction="none").sum(1).mean() * (temperature ** 2)

    # rearrange original
    re_logits = rank_reverse(logits_origin)
    pred_nt_origin = F.softmax(re_logits / temperature - 1000.0 * gt_mask, dim=1)

    log_pred_nt_cheap = F.log_softmax(logits_cheap / temperature - 1000.0 * gt_mask, dim=1)
    negative_loss = F.kl_div(log_pred_nt_cheap, pred_nt_origin, reduction="none").sum(1).mean() * (temperature ** 2)

    loss = alpha * positive_loss + beta * negative_loss
    # loss = beta * negative_loss
    return loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


if __name__ == "__main__":
    # ************************** set log **************************
    set_logger(os.path.join(args.save_path, 'training.log'))

    # #################### Load the parameters from json file #####################################
    json_path = os.path.join(args.save_path, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    params.cuda = torch.cuda.is_available() # use GPU if available

    for k, v in params.__dict__.items():
        logging.info('{}:{}'.format(k, v))

    # ########################################## Dataset ##########################################
    trainloader = fetch_dataloader('train', params)
    devloader = fetch_dataloader('dev', params)

    # ############################################ Model ############################################
    if params.dataset == 'cifar10':
        num_class = 10
    elif params.dataset == 'cifar100':
        num_class = 100
    elif params.dataset == 'tiny_imagenet':
        num_class = 200
    else:
        num_class = 10

    logging.info('Number of class: ' + str(num_class))

    logging.info('Create Model --- ' + params.model_name)

    # ResNet 18 / 34 / 50 ****************************************
    if params.model_name == 'resnet8':
        model = resnet8(num_classes=num_class)
    elif params.model_name == 'resnet18':
        model = ResNet18(num_class=num_class)
    elif params.model_name == 'resnet34':
        model = ResNet34(num_class=num_class)
    elif params.model_name == 'resnet50':
        model = ResNet50(num_class=num_class)

    #  vgg family
    elif params.model_name == 'vgg8':
        model = vgg8_bn(num_classes=num_class)
    elif params.model_name == 'vgg13':
        model = vgg13_bn(num_classes=num_class)

    # wrn family
    elif params.model_name == 'wrn16':
        model = wrn_16_2(num_classes=num_class)
    elif params.model_name == 'wrn40':
        model = wrn_40_2(num_classes=num_class)

    # PreResNet(ResNet for CIFAR-10)  20/32/56/110 ***************
    elif params.model_name.startswith('preresnet20'):
        model = PreResNet(depth=20, num_classes=num_class)
    elif params.model_name.startswith('preresnet32'):
        model = PreResNet(depth=32, num_classes=num_class)
    elif params.model_name.startswith('preresnet44'):
        model = PreResNet(depth=44, num_classes=num_class)
    elif params.model_name.startswith('preresnet56'):
        model = PreResNet(depth=56, num_classes=num_class)
    elif params.model_name.startswith('preresnet110'):
        model = PreResNet(depth=110, num_classes=num_class)

    # DenseNet *********************************************
    elif params.model_name == 'densenet121':
        model = densenet121(num_class=num_class)
    elif params.model_name == 'densenet161':
        model = densenet161(num_class=num_class)
    elif params.model_name == 'densenet169':
        model = densenet169(num_class=num_class)

    # ResNeXt *********************************************
    elif params.model_name == 'resnext29':
        model = CifarResNeXt(cardinality=8, depth=29, num_classes=num_class)

    elif params.model_name == 'mobilenetv2':
        model = MobileNetV2(class_num=num_class)

    elif params.model_name == 'shufflenetv2':
        model = shufflenetv2(class_num=num_class)

    # Basic neural network ********************************
    elif params.model_name == 'net':
        model = Net(num_class, params)

    elif params.model_name == 'mlp':
        model = MLP(num_class=num_class)

    else:
        model = None
        print('Not support for model ' + str(params.model_name))
        exit()

    # Adversarial model *************************************************************
    logging.info('Create Adversarial Model --- ' + params.adversarial_model)

    # ResNet 18 / 34 / 50 ****************************************
    if params.adversarial_model == 'resnet18':
        adversarial_model = ResNet18(num_class=num_class)
    elif params.adversarial_model == 'resnet34':
        adversarial_model = ResNet34(num_class=num_class)
    elif params.adversarial_model == 'resnet50':
        adversarial_model = ResNet50(num_class=num_class)

    #  vgg family
    elif params.adversarial_model == 'vgg8':
        adversarial_model = vgg8_bn(num_classes=num_class)
    elif params.adversarial_model == 'vgg13':
        adversarial_model = vgg13_bn(num_classes=num_class)

    # wrn family
    elif params.adversarial_model == 'wrn16':
        adversarial_model = wrn_16_2(num_classes=num_class)
    elif params.adversarial_model == 'wrn40':
        adversarial_model = wrn_40_2(num_classes=num_class)

    # PreResNet(ResNet for CIFAR-10)  20/32/56/110 ***************
    elif params.adversarial_model.startswith('preresnet20'):
        adversarial_model = PreResNet(depth=20)
    elif params.adversarial_model.startswith('preresnet32'):
        adversarial_model = PreResNet(depth=32)
    elif params.adversarial_model.startswith('preresnet56'):
        adversarial_model = PreResNet(depth=56)
    elif params.adversarial_model.startswith('preresnet110'):
        adversarial_model = PreResNet(depth=110)

    # DenseNet *********************************************
    elif params.adversarial_model == 'densenet121':
        adversarial_model = densenet121(num_class=num_class)
    elif params.adversarial_model == 'densenet161':
        adversarial_model = densenet161(num_class=num_class)
    elif params.adversarial_model == 'densenet169':
        adversarial_model = densenet169(num_class=num_class)

    # ResNeXt *********************************************
    elif params.adversarial_model == 'resnext29':
        adversarial_model = CifarResNeXt(cardinality=8, depth=29, num_classes=num_class)

    elif params.adversarial_model == 'mobilenetv2':
        adversarial_model = MobileNetV2(class_num=num_class)

    elif params.adversarial_model == 'shufflenetv2':
        adversarial_model = shufflenetv2(class_num=num_class)

    # Basic neural network ********************************
    elif params.adversarial_model == 'net':
        adversarial_model = Net(num_class, params)

    elif params.adversarial_model == 'mlp':
        adversarial_model = MLP(num_class=num_class)

    else:
        adversarial_model = None
        print('Not support for model ' + str(params.adversarial_model))
        exit()

    if params.cuda:
        model = model.cuda()
        adversarial_model = adversarial_model.cuda()

    if len(args.gpu_id) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        adversarial_model = nn.DataParallel(adversarial_model, device_ids=device_ids)

    # checkpoint ********************************
    if args.resume:
        logging.info('- Load checkpoint from {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logging.info('- Finetune training. ')
        checkpoint = torch.load(params.adversarial_resume)
        model.load_state_dict(checkpoint['state_dict'])

    # load trained Adversarial model ****************************
    logging.info('- Load Trained adversarial model from {}'.format(params.adversarial_resume))
    checkpoint = torch.load(params.adversarial_resume)
    adversarial_model.load_state_dict(checkpoint['state_dict'])

    # ############################### Optimizer ###############################
    if params.model_name == 'net' or params.model_name == 'mlp':
        optimizer = Adam(model.parameters(), lr=params.learning_rate)
        logging.info('Optimizer: Adam')
    else:
        optimizer = SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
        logging.info('Optimizer: SGD')

    # ************************** train and evaluate **************************
    train_and_eval_kd_adv(model, adversarial_model, optimizer, trainloader, devloader, params)

