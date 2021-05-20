import math

from torch import nn
from tqdm import tqdm
import torch
from config.config import *
from layers.hard_example import *
from layers.lovasz_losses import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve

class WeightedL1(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32).cuda()

    def forward(self, y_pred, y_true, epoch=1):
        #size = list(y_pred.size())[0]
        vals = y_pred.cuda() - y_true.to(torch.float).cuda()
        vals = torch.matmul(vals, self.weights)
        vals = torch.abs(vals)
        vals = vals.mean()
        return vals

class L1(nn.Module):
    def __init__(self):
        super().__init__()
    def name(self):
        return 'L1'

    def forward(self, y_pred, y_true, epoch=1):
        #size = list(y_pred.size())[0]
        vals = y_pred.cuda() - y_true.to(torch.float).cuda()
        vals = torch.abs(vals)
        vals = vals.mean()
        return vals

class WeightedL2(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32).cuda()

    def forward(self, y_pred, y_true, epoch=1):
        #size = list(y_pred.size())[0]
        vals = y_pred.cuda() - y_true.to(torch.float).cuda()
        vals = torch.mul(vals, vals)
        vals = torch.matmul(vals, self.weights)
        vals = vals.mean()
        return vals

class ROCStar(nn.Module):
    def __init__(self, delta, train_loader):
        super().__init__()
        self.delta = delta
        tmp = np.ones(len(LABEL_NAMES)) * 0.2
        self.default_gamma = torch.tensor(tmp, dtype=torch.float).cuda() # Starting value, will update each epoch
        self.gamma = self.default_gamma
        self.classnum = torch.tensor(len(LABEL_NAMES), dtype=torch.float).cuda()

        # Initialize last epoch with random values
        for o in tqdm(train_loader):
            if 'q' in locals():
                q = torch.cat((q, o[1]))
            else:
                q = o[1]

        self.epoch_true = q.cuda()
        h, w = list(self.epoch_true.size())
        initvals = np.random.rand(h, w)
        initvals = (1.0 - initvals) / 2.0
        self.epoch_pred = torch.tensor(initvals, dtype=torch.float).cuda()

    def name(self):
        return 'ROCStar'

    def forward(self, y_pred, y_true, epoch=1):
        loss = torch.tensor(0.0, dtype=torch.float).cuda()
        loss.requires_grad = True
        for i in range(0, len(LABEL_NAMES)):
            loss = torch.add(loss, self.roc_star_loss(y_true[:,i], y_pred[:,i], self.gamma[i], self.epoch_true[:,i], self.epoch_pred[:,i]))

        loss = torch.div(loss, self.classnum)
        return loss

    def update_on_epoch_end(self, epoch_pred, epoch_true, epoch=-1):
        self.epoch_pred = epoch_pred
        self.epoch_true = epoch_true

        for i in range(0, len(LABEL_NAMES)):
            self.gamma[i] = self._epoch_update_gamma(self.epoch_pred[:,i], self.epoch_true[:,i], epoch, self.delta)
            self.gamma = self.gamma.type(torch.FloatTensor)

    def roc_star_loss(self, _y_true, y_pred, gamma, _epoch_true, epoch_pred):
        """
        Nearly direct loss function for AUC.
        See article,
        C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
        https://github.com/iridiumblue/articles/blob/master/roc_star.md
            _y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
            y_pred: `Tensor` . Predictions.
            gamma  : `Float` Gamma, as derived from last epoch.
            _epoch_true: `Tensor`.  Targets (labels) from last epoch.
            epoch_pred : `Tensor`.  Predicions from last epoch.
        """
        #convert labels to boolean
        y_true = (_y_true>=0.50)
        epoch_true = (_epoch_true>=0.50)

        # if batch is either all true or false return small random stub value.
        if torch.sum(y_true)==0 or torch.sum(y_true) == y_true.shape[0]: return torch.sum(y_pred)*1e-8

        pos = y_pred[y_true]
        neg = y_pred[~y_true]

        epoch_pos = epoch_pred[epoch_true]
        epoch_neg = epoch_pred[~epoch_true]

        # Take random subsamples of the training set, both positive and negative.
        max_pos = 1000 # Max number of positive training samples
        max_neg = 1000 # Max number of positive training samples
        cap_pos = epoch_pos.shape[0]
        cap_neg = epoch_neg.shape[0]
        epoch_pos = epoch_pos[torch.rand_like(epoch_pos) < max_pos/cap_pos]
        epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg/cap_pos]

        ln_pos = pos.shape[0]
        ln_neg = neg.shape[0]

        # sum positive batch elements against (subsampled) negative elements
        if ln_pos>0 :
            pos_expand = pos.view(-1,1).expand(-1,epoch_neg.shape[0]).reshape(-1)
            neg_expand = epoch_neg.repeat(ln_pos)

            diff2 = neg_expand.type(torch.FloatTensor).cuda() - pos_expand.type(torch.FloatTensor).cuda() + gamma.type(torch.FloatTensor).cuda()
            l2 = diff2[diff2>0]
            m2 = l2 * l2
            len2 = l2.shape[0]
        else:
            m2 = torch.tensor([0], dtype=torch.float).cuda()
            len2 = 0

        # Similarly, compare negative batch elements against (subsampled) positive elements
        if ln_neg>0 :
            pos_expand = epoch_pos.view(-1,1).expand(-1, ln_neg).reshape(-1)
            neg_expand = neg.repeat(epoch_pos.shape[0])

            diff3 = neg_expand.type(torch.FloatTensor).cuda() - pos_expand.type(torch.FloatTensor).cuda() + gamma.type(torch.FloatTensor).cuda()
            l3 = diff3[diff3>0]
            m3 = l3*l3
            len3 = l3.shape[0]
        else:
            m3 = torch.tensor([0], dtype=torch.float).cuda()
            len3=0

        if (torch.sum(m2)+torch.sum(m3))!=0 :
           res2 = torch.sum(m2)/max_pos+torch.sum(m3)/max_neg
           #code.interact(local=dict(globals(), **locals()))
        else:
           res2 = torch.sum(m2)+torch.sum(m3)

        res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

        return res2

    def _epoch_update_gamma(self, y_pred, y_true, epoch=-1, delta=2):
        """
        Calculate gamma from last epoch's targets and predictions.
        Gamma is updated at the end of each epoch.
        y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
        y_pred: `Tensor` . Predictions.
        """
        DELTA = self.delta
        SUB_SAMPLE_SIZE = 2000.0
        pos = y_pred[y_true == 1]
        neg = y_pred[y_true == 0]  # yo pytorch, no boolean tensors or operators?  Wassap?
        # subsample the training set for performance
        cap_pos = pos.shape[0]
        cap_neg = neg.shape[0]
        pos = pos[torch.rand_like(pos) < SUB_SAMPLE_SIZE / cap_pos]
        neg = neg[torch.rand_like(neg) < SUB_SAMPLE_SIZE / cap_neg]
        ln_pos = pos.shape[0]
        ln_neg = neg.shape[0]
        pos_expand = pos.view(-1, 1).expand(-1, ln_neg).reshape(-1)
        neg_expand = neg.repeat(ln_pos)
        diff = neg_expand - pos_expand
        ln_All = diff.shape[0]
        Lp = diff[diff > 0]  # because we're taking positive diffs, we got pos and neg flipped.
        ln_Lp = Lp.shape[0] - 1
        diff_neg = -1.0 * diff[diff < 0]
        diff_neg = diff_neg.sort()[0]
        ln_neg = diff_neg.shape[0] - 1
        ln_neg = max([ln_neg, 0])
        left_wing = int(ln_Lp * DELTA)
        left_wing = max([0, left_wing])
        left_wing = min([ln_neg, left_wing])

        if diff_neg.shape[0] > 0:
            gamma = diff_neg[left_wing]
        else:
            gamma = self.default_gamma  # default=torch.tensor(0.2, dtype=torch.float).cuda() #zoink
        L1 = diff[diff > -1.0 * gamma]
        ln_L1 = L1.shape[0]

        if epoch > -1:
            return gamma.float()
        else:
            return self.default_gamma

class L2(nn.Module):
    def __init__(self):
        super().__init__()
    def name(self):
        return 'L2'

    def forward(self, y_pred, y_true, epoch=1):
        #size = list(y_pred.size())[0]
        vals = y_pred.cuda() - y_true.to(torch.float).cuda()
        vals = torch.mul(vals, vals)
        vals = vals.mean()
        return vals

class NeoFocalLoss(nn.Module):
    def __init__(self, gamma=2, weights=None):
        super().__init__()
        self.gamma = gamma
        if weights is None:
            weights = torch.tensor([1.0, 16.0, 5.0, 13.0, 9.0, 9.0, 13.0, 5.0, 30.0, 18.0, 19.0, 100.0, 10.0, 3.0, 4.0, 50.0, 1.0, 3.0, 6.0]).cuda()
        self.class_weights = weights
        self.weightsum = torch.sum(self.class_weights).cuda()

    def forward(self, logit, target, epoch=0):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        loss = loss * self.class_weights
        loss = loss / self.weightsum

        if len(loss.size())==2:
            loss = loss.sum(dim=1)

        return loss.mean()

    def name(self):
        return 'NeoFocalLoss'


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target, epoch=0):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)

        return loss.mean()

    def name(self):
        return 'FocalLoss'

class WeightedFocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class HardLogLoss(nn.Module):
    def __init__(self):
        super(HardLogLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.__classes_num = NUM_CLASSES

    def name(self):
        return 'HardLogLoss'

    def forward(self, logits, labels, epoch=0):
        labels = labels.float()
        loss=0
        for i in range(labels.shape[1]):
            logit_ac=logits[:,i]
            label_ac=labels[:,i]
            logit_ac, label_ac=get_hard_samples(logit_ac,label_ac)
            loss+=self.bce_loss(logit_ac,label_ac)
        loss = loss/labels.shape[1]
        return loss

# https://github.com/bermanmaxim/LovaszSoftmax/tree/master/pytorch
def lovasz_hinge(logits, labels, ignore=None, per_class=True):
    """
    Binary Lovasz hinge loss
      logits: [B, C] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, C] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_class:
        loss = 0
        for i in range(NUM_CLASSES):
            logit_ac = logits[:, i]
            label_ac = labels[:, i]
            loss += lovasz_hinge_flat(logit_ac, label_ac)
        loss = loss / NUM_CLASSES
    else:
        logits = logits.view(-1)
        labels = labels.view(-1)
        loss = lovasz_hinge_flat(logits, labels)
    return loss

# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69053
class SymmetricLovaszLoss(nn.Module):
    def __init__(self):
        super(SymmetricLovaszLoss, self).__init__()
        self.__classes_num = NUM_CLASSES

    def forward(self, logits, labels,epoch=0):
        labels = labels.float()
        loss=((lovasz_hinge(logits, labels)) + (lovasz_hinge(-logits, 1 - labels))) / 2
        return loss

class FocalSymmetricLovaszHardLogLoss(nn.Module):
    def __init__(self):
        super(FocalSymmetricLovaszHardLogLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.slov_loss = SymmetricLovaszLoss()
        self.log_loss = HardLogLoss()
    def forward(self, logit, labels,epoch=0):
        labels = labels.float()
        focal_loss = self.focal_loss.forward(logit, labels, epoch)
        slov_loss = self.slov_loss.forward(logit, labels, epoch)
        log_loss = self.log_loss.forward(logit, labels, epoch)
        loss = focal_loss*0.5 + slov_loss*0.5 +log_loss * 0.5
        return loss

class mAP(nn.Module):
    def __init__(self):
        super(mAP, self).__init__()
    def forward(self, logit, labels,epoch=0):
        labels = labels.float().cpu().detach().numpy()
        preds = logit.float().cpu().detach().numpy()

        auc_vals = []
        for i in range(0, len(labels[0])):
            precision, recall, thresholds = precision_recall_curve(labels[:,i], preds[:,i])
            # Use AUC function to calculate the area under the curve of precision recall curve
            area = auc(recall, precision)
            if math.isnan(area):
                area = 0.0
            auc_vals.append(area)
        auc_vals = np.stack(auc_vals)
        return np.average(auc_vals)

# https://github.com/ronghuaiyang/arcface-pytorch
class ArcFaceLoss(nn.modules.Module):
    def __init__(self,s=30.0,m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels, epoch=0):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss1 = self.classify_loss(output, labels)
        loss2 = self.classify_loss(cosine, labels)
        gamma=1
        loss=(loss1+gamma*loss2)/(1+gamma)
        return loss
