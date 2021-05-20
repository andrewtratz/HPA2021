# coding: utf-8
import sys
sys.path.insert(0, '')
import argparse

import time
import shutil
import pandas as pd

import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.cuda.amp import autocast

from utils.puzzle_utils import *
from utils.common_util import *
from utils.torch_utils import *
from networks.imageclsnet import init_network
from datasets.protein_dataset import ProteinDataset
from networks.densenet import Identity
from utils.augment_util import train_multi_augment2, puzzle_augment
from layers.loss import *
from utils.log_util import Logger

loss_names = ['FocalSymmetricLovaszHardLogLoss']
split_names = ['random_ext_folds5', 'random_ext_noleak_clean_folds5']

parser = argparse.ArgumentParser(description='PyTorch Protein Classification')
parser.add_argument('--out_dir', type=str, help='destination where trained network should be saved')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id used for training (default: 0)')
parser.add_argument('--arch', default='class_densenet121_dropout', type=str,
                    help='model architecture (default: class_densenet121_dropout)')
parser.add_argument('--num_classes', default=28, type=int, help='number of classes (default: 28)')
parser.add_argument('--in_channels', default=4, type=int, help='in channels (default: 4)')
parser.add_argument('--scheduler', default='Adam45', type=str, help='scheduler name')
parser.add_argument('--epochs', default=55, type=int, help='number of total epochs to run (default: 55)')
parser.add_argument('--img_size', default=768, type=int, help='image size (default: 768)')
parser.add_argument('--crop_size', default=512, type=int, help='crop size (default: 512)')
parser.add_argument('--batch_size', default=32, type=int, help='train mini-batch size (default: 32)')
parser.add_argument('--effective_batch', default=32, type=int, help='train mini-batch size (default: 32)')
parser.add_argument('--workers', default=3, type=int, help='number of data loading workers (default: 3)')
parser.add_argument('--split_name', default='random_ext_folds5', type=str)
parser.add_argument('--fold', default=0, type=int, help='index of fold (default: 0)')
parser.add_argument('--clipnorm', default=1, type=int, help='clip grad norm')
parser.add_argument('--resume', default=None, type=str, help='name of the latest checkpoint (default: None)')
parser.add_argument('--puzzle', default=0, type=int, help='set to 1 to enable puzzle module')
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--alpha_schedule', default=0.50, type=float)
parser.add_argument('--labelmask', default='', type=str)
parser.add_argument('--dataset', default='train', type=str)
parser.add_argument('--sample', default='False')
parser.add_argument('--loss', default='BCE')
parser.add_argument('--cell', default='')
parser.add_argument('--mixed_precision', default='True')
parser.add_argument('--resetscale', default='False')
parser.add_argument('--preload', default='bestfitting')
parser.add_argument('--result_dir', default=RESULT_DIR)
parser.add_argument('--img_path', default='')

args = parser.parse_args()
# set cuda visible device
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def main():

    print(torch.cuda.device_count())

    device_ids = list(map(int, args.gpu_id.split(',')))

    if args.resetscale == 'True':
        RESETSCALE = True
    else:
        RESETSCALE = False

    CELL = args.cell
    if CELL == 'nuclei':
        NUCLEI = True
    else:
        NUCLEI = False

    if args.mixed_precision == 'True':
        mixed_precision = True
    else:
        mixed_precision = False

    if len(args.gpu_id.split(',')) > 1:
        PARALLEL = True
        device = torch.device('cuda', 0)
    else:
        PARALLEL = False
        device = torch.device('cuda', int(args.gpu_id))

    if args.sample == 'True':
        SAMPLE = True
    else:
        SAMPLE = False

    if args.puzzle == 1:
        PUZZLE = True
    else:
        PUZZLE = False

    RESULT_DIR = args.result_dir

    DATASET = args.dataset

    if args.img_path == '':
        IMG_PATH = IMG_PATHS[DATASET]
    else:
        IMG_PATH = args.img_path

    # Are we resuming from bestfitting or ourselves?
    if args.preload == 'bestfitting':
        PRELOAD_BF = True
    else:
        PRELOAD_BF = False

    log_out_dir = opj(RESULT_DIR, 'logs', args.out_dir, 'fold%d' % args.fold)
    if not ope(log_out_dir):
        os.makedirs(log_out_dir)
    log = Logger()
    log.open(opj(log_out_dir, 'log.train.txt'), mode='a')

    # Data loading code
    if PUZZLE:
        train_transform = puzzle_augment
    else:
        train_transform = train_multi_augment2


    if SAMPLE:
        train_split_file = opj(DATA_DIR, 'split', args.split_name, 'random_train_cv%d sample.csv' % args.fold)
    else:
        if CELL == 'nuclei':
            train_split_file = opj('F:\\nuclei_meta_train.csv')
        elif args.cell == 'cell':
            #train_split_file = opj(DATA_DIR, 'split', args.split_name, 'random_train_cell_cv%d.csv' % args.fold)
            train_split_file = opj('.', 'random_train_cell_cv%d.csv' % args.fold)
        else:
            train_split_file = opj('.', 'random_train_cv%d.csv' % args.fold)
    train_df = pd.read_csv(train_split_file)
    if args.cell == 'nuclei':
        labelset = train_df[NUCLEI_NAME_LIST].values
    else:
        labelset = train_df[LABEL_NAME_LIST].values

    if args.labelmask != '':
        labelmask = np.array(list(map(int, args.labelmask.split(','))))
    else:
        labelmask = np.ones(labelset.shape[1])

    train_dataset = ProteinDataset(
        train_split_file,
        img_size=args.img_size,
        is_trainset=True,
        return_label=True,
        in_channels=args.in_channels,
        transform=train_transform,
        crop_size=args.crop_size,
        random_crop=True,
        #puzzle=PUZZLE,
        labelmask=labelmask,
        nuclei=NUCLEI,
        img_dir=IMG_PATH
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    if SAMPLE:
        valid_split_file = opj(DATA_DIR, 'split', args.split_name, 'random_valid_cv%d sample.csv' % args.fold)
    else:
        if CELL == 'nuclei':
            valid_split_file = opj('F:\\nuclei_meta_cv.csv')
        elif CELL == 'cell':
            #valid_split_file = opj(DATA_DIR, 'split', args.split_name, 'random_valid_cell_cv%d.csv' % args.fold)
            valid_split_file = opj('.', 'random_valid_cell_cv%d.csv' % args.fold)
        else:
            valid_split_file = opj('.', 'random_valid_cv%d.csv' % args.fold)

    valid_dataset = ProteinDataset(
        valid_split_file,
        img_size=args.img_size,
        is_trainset=True,
        return_label=True,
        in_channels=args.in_channels,
        transform=None,
        crop_size=args.crop_size,
        random_crop=True,
        #puzzle=PUZZLE,
        labelmask=labelmask,
        nuclei=NUCLEI,
        img_dir=IMG_PATH
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True
    )



    ###############################
    # network_path = opj(RESULT_DIR, 'models', args.out_dir, 'fold%d' % args.fold, '%s.pth' % args.predict_epoch)
    ###############################

    model_out_dir = opj(RESULT_DIR, 'models', args.out_dir, 'fold%d' % args.fold)
    log.write(">> Creating directory if it does not exist:\n>> '{}'\n".format(model_out_dir))
    if not ope(model_out_dir):
        os.makedirs(model_out_dir)


    cudnn.benchmark = True

    # set random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    model_params = {}
    model_params['architecture'] = args.arch
    if PRELOAD_BF:
        model_params['num_classes'] = 28
    else:
        model_params['num_classes'] = args.num_classes
    model_params['in_channels'] = args.in_channels
    model = init_network(model_params)

    # move network to gpu
    # model = DataParallel(model)
    # model.cuda()

    # Only when preloading my own models
    if not PRELOAD_BF and 'efficientnet' not in args.arch:
        model.logit = nn.Linear(model.num_features, args.num_classes)
        model.backbone.features.classifier = Identity()
        model.backbone.features.norm5 = Identity()
        model.backbone.classifier = Identity()
        model.num_classes = args.num_classes

    start_epoch = 0
    best_loss = 1e5
    best_epoch = 0
    best_criterion = 0.0

    #Create GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    # optionally resume from a checkpoint
    if args.resume:
        #args.resume = os.path.join(model_out_dir, args.resume)
        if os.path.isfile(args.resume):
            # load checkpoint weights and update model and optimizer
            log.write(">> Loading checkpoint:\n>> '{}'\n".format(args.resume))

            checkpoint = torch.load(args.resume)
            start_epoch = 0
            best_epoch = -1
            best_criterion = 0.0
            if not PRELOAD_BF:
                start_epoch = checkpoint['epoch'] + 1
                best_epoch = checkpoint['best_epoch']
                best_criterion = checkpoint['best_score']

            model.load_state_dict(checkpoint['state_dict'])

            if PRELOAD_BF:
                # Replace final FC layer with correct class number
                model.logit = nn.Linear(model.num_features, args.num_classes)
                model.backbone.features.classifier = Identity()
                model.backbone.features.norm5 = Identity()
                model.backbone.classifier = Identity()
                model.num_classes = args.num_classes

            if PUZZLE:
                model.add_puzzle()

            # define scheduler
            try:
                scheduler = eval(args.scheduler)()
            except:
                raise (RuntimeError("Scheduler {} not available!".format(args.scheduler)))
            optimizer = scheduler.schedule(model, start_epoch, args.epochs)[0]

            if not RESETSCALE:
                optimizer_fpath = args.resume.replace('.pth', '_optim.pth')
                if ope(optimizer_fpath):
                    log.write(">> Loading optimizer:\n>> '{}'\n".format(optimizer_fpath))
                    optimizer.load_state_dict(torch.load(optimizer_fpath)['optimizer'])

                scaler_fpath = args.resume.replace('.pth', '_scaler.pth')
                if ope(scaler_fpath):
                    log.write(">> Loading scaler:\n>> '{}'\n".format(scaler_fpath))
                    scaler.load_state_dict(torch.load(scaler_fpath)['scaler'])

            log.write(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})\n".format(args.resume, checkpoint['epoch']))
        else:
            if PUZZLE:
                model.add_puzzle()

            if PARALLEL:
                model = torch.nn.DataParallel(model, device_ids=device_ids)
                model.cuda()
            else:
                model.cuda(device=device)
    else:
        # define scheduler
        try:
            scheduler = eval(args.scheduler)()
        except:
            raise (RuntimeError("Scheduler {} not available!".format(args.scheduler)))
        optimizer = scheduler.schedule(model, start_epoch, args.epochs)[0]
        log.write(">> No checkpoint found at '{}'\n".format(args.resume))

    if PARALLEL:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.cuda()
    else:
        model.cuda(device=device)

    #print(model)
    #model.freeze_backbone()

    lossfunct = eval(LOSSES[args.loss])
    log.write('Loss function: ' + LOSSES[args.loss] + '\n')
    criterion = mAP()

    if args.loss == 'ROCStar':
        log.write('Initializing ROCStar...')
        lossmod = lossfunct(delta=1.0, train_loader=train_loader)
        valid_lossmod = L2()
    else:
        lossmod = lossfunct()
        valid_lossmod = lossfunct()

    if PUZZLE:
        log.write('** start training here! **\n')
        log.write('\n')
        log.write('epoch    iter      rate     |  loss/class_loss/p_class_loss/re  |   alpha   |    valid_loss/focal/kaggle     |best_epoch/best_focal|  min \n')
        log.write('------------------------------------------------------------------------------------------------------------------------------------------\n')
    else:
        weights = get_labelweights(labelset)
        # weights = np.minimum(weights, 10.0)
        log.write('** start training here! **\n')
        log.write('\n')
        log.write('epoch    iter      rate     |  train_loss  |    val_loss  val_crit  |best_epoch/best_crit|  min \n')
        log.write('-----------------------------------------------------------------------------------------------------------------\n')

    start_epoch += 1

    for epoch in range(start_epoch, args.epochs + 1):
        end = time.time()

        # set manual seeds per epoch
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

        # adjust learning rate for each epoch
        lr_list = scheduler.step(model, epoch, args.epochs)
        lr = lr_list[0]

        # adjust alpha according to schedule
        alpha = min((args.alpha * epoch) / (args.epochs * args.alpha_schedule), args.alpha)

        # train for one epoch on train set
        skipbatch = args.effective_batch // args.batch_size

        iteration, lossdict = train(train_loader=train_loader, model=model, lossmod=lossmod, optimizer=optimizer,
                                    epoch=epoch, clipnorm=args.clipnorm, lr=lr, batch_size=args.batch_size,
                                    skipbatch=skipbatch, puzzle=PUZZLE, alpha=alpha, device=device, scaler=scaler, mixed_precision=mixed_precision)
        with torch.no_grad():
            valid_loss, valid_criterion = validate(valid_loader, model, valid_lossmod, epoch, criterion, device=device)

        # remember best loss and save checkpoint
        is_best = valid_criterion > best_criterion
        best_loss = min(valid_loss, best_loss)
        best_epoch = epoch if is_best else best_epoch
        best_criterion = valid_criterion if is_best else best_criterion


        if PUZZLE:
            print('\r', end='', flush=True)
            log.write(
                '%5.1f   %5d    %0.6f   |  %0.4f  %0.4f  %0.4f  |  %0.4f   |    %0.4f  %6.4f %6.4f       |  %6.1f    %6.4f   | %3.1f min \n' % \
                (epoch, iteration + 1, lr, lossdict['train'], lossdict['class'], lossdict['pclass'], lossdict['re'],
                 lossdict['alpha'], valid_loss, valid_criterion, best_epoch, best_criterion, (time.time() - end) / 60))
        else:
            print('\r', end='', flush=True)
            log.write(
                '%5.1f   %5d    %0.6f   |    %0.4f    |     %0.4f   %0.4f    | %6.1f    %6.4f   | %3.1f min \n' % \
                (epoch, iteration + 1, lr, lossdict['train'], valid_loss, valid_criterion, best_epoch, best_criterion,
                 (time.time() - end) / 60))



        save_model(model, is_best, model_out_dir, optimizer=optimizer, scaler=scaler, epoch=epoch, best_epoch=best_epoch, best_focal=best_criterion)

def train(train_loader, model, lossmod, optimizer, epoch, clipnorm=1, lr=1e-5, batch_size=32, skipbatch=1,
          puzzle=False, alpha=1.0, device=None, scaler=None, mixed_precision=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    class_losses = AverageMeter()
    p_class_losses = AverageMeter()
    re = AverageMeter()
    #lossmod = lossfunct()
    lossdict = {}

    if device is None:
        device = torch.cuda.current_device()

    if lossmod.name() == 'ROCStar':
        whole_y_pred = np.empty((0, len(LABEL_NAMES)), dtype='float')
        whole_y_t = np.empty((0, len(LABEL_NAMES)), dtype='float')

    # switch to train mode
    model.train()

    num_its = len(train_loader)
    end = time.time()
    print_freq = 1
    optimizer.zero_grad()
    for iteration, iter_data in enumerate(train_loader, 0):
        # measure data loading time
        data_time.update(time.time() - end)

        images, labels, indices = iter_data

        with autocast(mixed_precision): # Use mixed precision

            images = Variable(images.cuda(device))
            labels = Variable(labels.cuda(device)).float()

            logits, features = model(images)

            class_loss = lossmod.forward(logits, labels)

            #############################
            # Puzzle Module
            #############################
            if puzzle:
                tiled_images = tile_features(images, 4) # split into quadrants

                tiled_logits, tiled_features = model(tiled_images)

                re_features = merge_features(tiled_features, 4, batch_size)

                # Make CAMs - disabled by default in current SOTA
                # features = make_cam(features)
                # re_features = make_cam(re_features)

                # Pcl loss
                p_class_loss = lossmod(global_average_pooling_2d(x=re_features), labels, epoch=epoch)

                # RE loss with masking
                class_mask = labels.unsqueeze(2).unsqueeze(3)
                re_loss = L1_Loss(features, re_features) * class_mask
                re_loss = re_loss.mean()

                # Combined loss
                loss = class_loss + p_class_loss + alpha * re_loss
            else:
                loss = class_loss

            losses.update(loss.item())
            if puzzle:
                class_losses.update(class_loss.item())
                p_class_losses.update(p_class_loss.item())
                re.update(re_loss.item())

        # Exit mixed precision context before backward pass
        scaler.scale(loss).backward()

        if iteration % skipbatch == 0:
            #torch.nn.utils.clip_grad_norm(model.parameters(), clipnorm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if lossmod.name() == 'ROCStar':
            whole_y_pred = np.concatenate((whole_y_pred, F.sigmoid(logits).clone().detach().cpu().numpy().reshape(-1, len(LABEL_NAMES))))
            whole_y_t = np.concatenate((whole_y_t, labels.clone().detach().cpu().numpy().reshape(-1, len(LABEL_NAMES))))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #probs = F.sigmoid(logits)
        #acc = multi_class_acc(probs, labels)
        #accuracy.update(acc.item())

        if (iteration + 1) % print_freq == 0 or iteration == 0 or (iteration + 1) == num_its:
            if puzzle:
                print('\r%5.1f   %5d    %0.6f   |  %0.4f  %0.4f  %0.4f  %0.4f   |  %0.4f   | ... ' % \
                      (epoch - 1 + (iteration + 1) / num_its, iteration + 1, lr, losses.avg, class_losses.avg, p_class_losses.avg,
                       re.avg, alpha), end='', flush=True)
            else:
                print('\r%5.1f   %5d    %0.6f   |   %0.4f  ... ' % \
                      (epoch - 1 + (iteration + 1) / num_its, iteration + 1, lr, losses.avg), end='', flush=True)

    # Create dictionary of losses to pass back
    lossdict['train'] = losses.avg
    if puzzle:
        lossdict['class'] = class_losses.avg
        lossdict['pclass'] = p_class_losses.avg
        lossdict['re'] = re.avg
        lossdict['alpha'] = alpha

    if lossmod.name() == 'ROCStar':
        last_whole_y_t = torch.tensor(whole_y_t).cuda()
        last_whole_y_pred = torch.tensor(whole_y_pred).cuda()
        lossmod.update_on_epoch_end(last_whole_y_pred, last_whole_y_t, epoch)

    return iteration, lossdict

def validate(valid_loader, model, lossmod, epoch, criterion, threshold=0.5, device=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    if device is None:
        device = torch.cuda.current_device()

    # switch to evaluate mode
    model.eval()

    probs_list = []
    labels_list = []
    logits_list = []

    end = time.time()
    for it, iter_data in enumerate(valid_loader, 0):
        images, labels, indices = iter_data
        images = Variable(images.cuda(device))
        labels = Variable(labels.cuda(device)).float()

        with torch.no_grad():
            outputs, _ = model(images)
            loss = lossmod.forward(outputs, labels)

            logits = outputs
            probs = F.sigmoid(logits)
        #acc = multi_class_acc(probs, labels)

        probs_list.append(probs.cpu().detach().numpy())
        labels_list.append(labels.cpu().detach().numpy())
        logits_list.append(logits.cpu().detach().numpy())

        losses.update(loss.item())
        #accuracy.update(acc.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    probs = np.vstack(probs_list)
    y_true = np.vstack(labels_list)
    logits = np.vstack(logits_list)
    valid_criterion = criterion.forward(torch.from_numpy(logits), torch.from_numpy(y_true))

    y_pred = probs > threshold
    #kaggle_score = f1_score(y_true, y_pred, average='macro')

    return losses.avg, valid_criterion

def save_model(model, is_best, model_out_dir, optimizer=None, scaler=None, epoch=None, best_epoch=None, best_focal=None):
    if type(model) == DataParallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    model_fpath = opj(model_out_dir, '%03d.pth' % epoch)
    torch.save({
        'save_dir': model_out_dir,
        'state_dict': state_dict,
        'best_epoch': best_epoch,
        'epoch': epoch,
        'best_score': best_focal,
    }, model_fpath)

    optim_fpath = opj(model_out_dir, '%03d_optim.pth' % epoch)
    if optimizer is not None:
        torch.save({
            'optimizer': optimizer.state_dict(),
        }, optim_fpath)

    scaler_fpath = opj(model_out_dir, '%03d_scaler.pth' % epoch)
    if scaler is not None:
        torch.save({
            'scaler': scaler.state_dict(),
        }, scaler_fpath)

    if is_best:
        best_model_fpath = opj(model_out_dir, 'final.pth')
        shutil.copyfile(model_fpath, best_model_fpath)
        if optimizer is not None:
            best_optim_fpath = opj(model_out_dir, 'final_optim.pth')
            shutil.copyfile(optim_fpath, best_optim_fpath)

def multi_class_acc(preds, targs, th=0.5):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds == targs).float().mean()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_labelweights(labelset):
    counts = np.sum(labelset, axis=0).astype('float')
    total = float(np.max(counts))
    return np.divide(total, counts)

if __name__ == '__main__':
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    main()
    print('\nsuccess!')


