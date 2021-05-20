# coding: utf-8
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, '')
import argparse
from tqdm import tqdm
import pandas as pd

import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from torch.autograd import Variable
from torch.backends import cudnn
from torch.cuda.amp import autocast

from networks.densenet import Identity
from utils.common_util import *
from networks.imageclsnet import init_network
from datasets.protein_dataset import ProteinDataset
from utils.augment_util import *
from utils.log_util import Logger
from utils.torch_utils import *

datasets_names = ['test', 'val']
split_names = ['random_ext_folds5', 'random_ext_noleak_clean_folds5']
augment_list = ['default', 'flipud', 'fliplr','transpose', 'flipud_lr',
                'flipud_transpose', 'fliplr_transpose', 'flipud_lr_transpose']

parser = argparse.ArgumentParser(description='PyTorch Protein Classification')
parser.add_argument('--base_dir', default='', type=str, help='root file location')
parser.add_argument('--out_dir', type=str, help='destination where predicted result should be saved')
parser.add_argument('--network_path', default='', type=str, help='location of model file')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id used for predicting (default: 0)')
parser.add_argument('--arch', default='class_densenet121_dropout', type=str,
                    help='model architecture (default: class_densenet121_dropout)')
parser.add_argument('--num_classes', default=19, type=int, help='number of classes (default: 19)')
parser.add_argument('--in_channels', default=4, type=int, help='in channels (default: 4)')
parser.add_argument('--img_size', default=768, type=int, help='image size (default: 768)')
parser.add_argument('--crop_size', default=512, type=int, help='crop size (default: 512)')
parser.add_argument('--batch_size', default=32, type=int, help='train mini-batch size (default: 32)')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 0)')
parser.add_argument('--fold', default=0, type=int, help='index of fold (default: 0)')
parser.add_argument('--augment', default='default', type=str, help='test augmentation (default: default)')
parser.add_argument('--seed', default=100, type=int, help='random seed (default: 100)')
parser.add_argument('--seeds', default=None, type=str, help='predict seed')
parser.add_argument('--dataset', default='test', type=str)
parser.add_argument('--split_name', default='random_ext_folds5', type=str)
parser.add_argument('--predict_epoch', default=None, type=int, help='number epoch to predict')
parser.add_argument('--img_dir', default=None, type=str, help='directory of image files')
parser.add_argument('--puzzle', default='0', type=str, help='are we loading a puzzle model?')
parser.add_argument('--cellorimage', default='image')
parser.add_argument('--BGR', default='False')

class Arguments():
    def __init__(self):
        self.base_dir = ''
        self.out_dir = ''
        self.network_path = ''
        self.gpu_id = '0'
        self.arch = 'class_densenet121_dropout'
        self.num_classes = 19
        self.in_channels = 4
        self.img_size = 768
        self.crop_size = 512
        self.batch_size = 32
        self.workers = 0
        self.fold = 0
        self.augment = 'default'
        self.seed = 100
        self.seeds = None
        self.dataset = 'test'
        self.split_name = 'random_ext_folds5'
        self.predict_epoch = None
        self.img_dir = None
        self.puzzle = '0'
        self.cellorimage = 'image'
        self.BGR = 'False'

def inference(args=None, test_split_file=None):

    # Make sure we have CUDNN optimization turned on
    cudnn.benchmark = True
    cudnn.enabled = True

    if args == None:
        args = parser.parse_args()

    if args.BGR == 'True' or args.BGR == '1':
        BGR = True
    else:
        BGR = False

    cellorimage = args.cellorimage
    network_path = args.network_path

    if args.puzzle == '1':
        PUZZLE = True
    else:
        PUZZLE = False

    if args.base_dir == '':
        ROOT = RESULT_DIR
    else:
        ROOT = args.base_dir

    log_out_dir = opj(ROOT, 'logs', args.out_dir, 'fold%d' % args.fold)
    if not ope(log_out_dir):
        os.makedirs(log_out_dir)
    log = Logger()
    log.open(opj(log_out_dir, 'log.submit.txt'), mode='a')

    args.predict_epoch = 'final' if args.predict_epoch is None else '%03d' % args.predict_epoch

    if args.network_path == '':
        network_path = opj(ROOT, 'models', args.out_dir, 'fold%d' % args.fold, '%s.pth' % args.predict_epoch)

    submit_out_dir = opj(ROOT, 'submissions', args.out_dir, 'fold%d' % args.fold, 'epoch_%s' % args.predict_epoch)
    log.write(">> Creating directory if it does not exist:\n>> '{}'\n".format(submit_out_dir))
    if not ope(submit_out_dir):
        os.makedirs(submit_out_dir)

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    args.augment = args.augment.split(',')
    if args.augment[0] == 'all':
        args.augment = augment_list
    else:
        for augment in args.augment:
            if augment not in augment_list:
                raise ValueError('Unsupported or unknown test augmentation: {}!'.format(augment))

    model_params = {}
    model_params['architecture'] = args.arch
    model_params['num_classes'] = args.num_classes
    model_params['in_channels'] = args.in_channels
    model_params['pretrain'] = False
    if 'efficient' in args.arch:
        model_params['pretrain'] = True
    model = init_network(model_params)

    # Getting rid of extraneous features
    if 'efficient' not in args.arch:
        model.backbone.features.norm5 = Identity()
        model.backbone.classifier = Identity()
    else:
        model.backbone.model._fc = torch.nn.Linear(1280, 7)
        #model.backbone.model._conv_stem = torch.nn.Conv2d()


    if PUZZLE:
        model.add_puzzle()

    log.write(">> Loading network:\n>>>> '{}'\n".format(network_path))
    checkpoint = torch.load(network_path)
    model.load_state_dict(checkpoint['state_dict'])
    log.write(">>>> loaded network:\n>>>> epoch {}\n".format(checkpoint['epoch']))

    # moving network to gpu and eval mode
    #model = DataParallel(model)
    model.cuda()
    model.eval()

    # Data loading code
    dataset = args.dataset

    # Use test split file if provided as input, otherwise load from disk
    if test_split_file is None:
        if dataset == 'test':
            if cellorimage == 'cell':
                test_split_file = r'F:\public_cell_sample_submission.csv'
            else:
                #test_split_file = opj(DATA_DIR, 'split', 'test_11702.csv')
                test_split_file = r'D:\HPA\sample_submission.csv'
        elif dataset == 'val':
            test_split_file = opj(DATA_DIR, 'split', args.split_name, 'random_valid_cv%d.csv' % args.fold)
        elif dataset == 'custom':
            test_split_file = r'X:\bf_sample_submission.csv'
        elif dataset == 'custom512':
            test_split_file = r'X:\custom512_sample_submission.csv'
        else:
            raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))

    test_dataset = ProteinDataset(
        test_split_file,
        img_size=args.img_size,
        is_trainset=(dataset == 'train'),
        return_label=False,
        in_channels=args.in_channels,
        transform=None,
        crop_size=args.crop_size,
        random_crop=False,
        BGR=BGR,
        #puzzle=PUZZLE,
        img_dir=args.img_dir
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True,
    )


    seeds = [args.seed] if args.seeds is None else [int(i) for i in args.seeds.split(',')]
    for seed in seeds:
        test_dataset.random_crop = (seed != 0)
        for augment in args.augment:
            test_loader.dataset.transform = eval('augment_%s' % augment)
            if args.crop_size > 0:
                sub_submit_out_dir = opj(submit_out_dir, '%s_seed%d' % (augment, seed))
            else:
                sub_submit_out_dir = opj(submit_out_dir, augment)
            if not ope(sub_submit_out_dir):
                os.makedirs(sub_submit_out_dir)
            with torch.no_grad():
                #evaluate(model, test_loader)
                predict(test_loader, model, sub_submit_out_dir, dataset, savefeatures=PUZZLE, augment=augment)
                torch.cuda.empty_cache()

def display(images, title):
    raw_image = images.cpu().numpy()
    raw_image = np.array(np.transpose(raw_image[0, :, :, :], (1, 2, 0))[:, :, 0:3] * 255, dtype='uint8')
    plt.imshow(raw_image)
    plt.title(title)
    plt.show()

def predict(test_loader, model, submit_out_dir, dataset, savefeatures=False, augment='default', cell_or_image='image'):
    all_probs = []
    all_features = []

    img_ids = np.array(test_loader.dataset.img_ids)
    for it, iter_data in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
        images, indices = iter_data

        # Debugging code
        # display(images, img_ids[len(all_probs)])

        with autocast():
            images = Variable(images.cuda(), volatile=True)
            outputs, features = model(images)

        # if we got features back
        if savefeatures:
            i, c, h, w = features.shape
            all_features += features.cpu().numpy().tolist()

        logits = outputs
        probs = F.sigmoid(logits).data
        all_probs += probs.cpu().numpy().tolist()

    #img_ids = img_ids[:len(all_probs)]
    all_probs = np.array(all_probs).reshape(len(img_ids), -1)
    if all_probs.shape[1] == 7: # Nuclei situation
        fill_probs = np.zeros((len(img_ids), 19), dtype='float')
        fill_probs[:, 0:6] = all_probs[:, 0:6]
        fill_probs[:, 11] = all_probs[:, 6]
    else:
        fill_probs = all_probs

    np.save(opj(submit_out_dir, 'prob_%s.npy' % dataset), np.concatenate((img_ids.reshape(len(img_ids), -1), fill_probs), axis=1))

    if savefeatures:
        all_features = np.array(all_features).reshape(len(img_ids), c, h, w)
        broad = np.repeat(img_ids, 16*16).reshape((len(img_ids), 1, 16, 16))

        # Reapply all augments to reverse them (only works for directly reversible examples)
        if augment != 'default':
            aug = eval('augment_%s' % augment)
            for i in range(0, len(img_ids)):
                for j in range(0, len(LABEL_NAMES)):
                    if 'transpose' in augment:
                        temp = aug(all_features[i, j, :, :].reshape((all_features.shape[2], all_features.shape[3], 1)))
                        all_features[i, j, :, :] = temp[:, :, 0]
                    else:
                        all_features[i, j, :, :] = aug(all_features[i, j, :, :])

        np.save(opj(submit_out_dir, 'feat_%s.npy' % dataset), np.concatenate((broad, all_features), axis=1))

    result_df = prob_to_result(all_probs, img_ids)
    result_df.to_csv(opj(submit_out_dir, 'results_%s.csv.gz' % dataset), index=False, compression='gzip')

def prob_to_result(probs, img_ids, th=0.5):
    probs = probs.copy()
    probs[np.arange(len(probs)), np.argmax(probs, axis=1)] = 1

    pred_list = []
    for line in probs:
        s = ' '.join(list([str(i) for i in np.nonzero(line > th)[0]]))
        pred_list.append(s)
    result_df = pd.DataFrame({ID: img_ids, PREDICTED: pred_list})
    return result_df

if __name__ == '__main__':
    print('%s: calling inference function ... \n' % os.path.basename(__file__))
    #torch.backends.cudnn.benchmark = True # Fixes CUDNN error
    inference()
    print('\nsuccess!')