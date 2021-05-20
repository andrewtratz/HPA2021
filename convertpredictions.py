import argparse
import math
from mAP import *
from scipy.stats import rankdata

parser = argparse.ArgumentParser(description='Reformat predictions and combine augmentations')

parser.add_argument('--dataset', default='test', type=str)
parser.add_argument('--augment', default='default', type=str)
parser.add_argument('--truncate', default=19)
parser.add_argument('--model', default='Puzzle512')
parser.add_argument('--epoch', default=17)
parser.add_argument('--cellorimage', default='image')
parser.add_argument('--seeds', default='1')
parser.add_argument('--rank', default=1, type=int, help='use rank-based ensembling')
parser.add_argument('--predictions_path', default='')
parser.add_argument('--meta_csv_path', default='')
parser.add_argument('--cell_meta_csv_path', default='')
parser.add_argument('--out_path', default='')
parser.add_argument('--add_datetime', default=1)
parser.add_argument('--batch_size', default=1)


class BF_Arguments():
    def __init__(self):
        self.dataset = 'test'
        self.augment = 'default'
        self.truncate = 19
        self.model = 'Puzzle512'
        self.epoch = 17
        self.cellorimage = 'image'
        self.seeds = '100'
        self.rank = 1
        self.meta_csv_path = ''
        self.cell_meta_csv_path = ''
        self.predictions_path = ''
        self.out_path = ''
        self.add_datetime = 1
        self.batch_size = 1

def reformat_preds(args=None):
    # Use passed arguments or get from command line
    if args == None:
        args = parser.parse_args()

    cell_or_image = args.cellorimage
    seeds = args.seeds.split(',')
    truncate = int(args.truncate)

    if args.rank == 1:
        RANK = True
    else:
        RANK = False

    BATCH_SIZE = int(args.batch_size)
    DEBUG = False
    DATASET = args.dataset
    AUGMENT = args.augment

    CLASS_NUM = 19

    epsilon = 1e-5

    if args.predictions_path == '':
        PREDICTION_BASE = 'F:\\result\\submissions\\'
    else:
        PREDICTION_BASE = args.predictions_path

    epoch = str(args.epoch)
    if len(epoch) == 1:
        epoch = '0' + epoch

    if AUGMENT == 'default':
        augments = ['default']
    elif AUGMENT == 'all':
        augments = ['default', 'flipud', 'fliplr', 'transpose', 'flipud_lr', 'flipud_transpose', 'fliplr_transpose',
                    'flipud_lr_transpose']
    else:
        augments = AUGMENT.split(',')
        # assert(JOBTYPE != 'features')

    aug_probset = []
    aug_feat = []
    for augment in augments:

        BF_PREDICTIONS = [
            os.path.join(PREDICTION_BASE, args.model, 'fold0', 'epoch_0' + epoch, augment + '_seed' + str(seeds[0]))]
        MASKS = [np.ones((CLASS_NUM), dtype=int)]

        if args.meta_csv_path == '' and cell_or_image == 'image':
            meta_csv_path = 'X:\\' + DATASET + '_sample_submission.csv'
        if cell_or_image == 'image' and args.cell_meta_csv_path == '':
            cell_meta_csv_path = r'F:\public_cell_sample_submission.csv'
        if args.meta_csv_path == '' and cell_or_image == 'cell':
            meta_csv_path = r'F:\public_cell_sample_submission.csv'
            cell_meta_csv_path = meta_csv_path
        if args.meta_csv_path != '':
            meta_csv_path = args.meta_csv_path
        if args.cell_meta_csv_path != '':
            cell_meta_csv_path = args.cell_meta_csv_path

        df_test = pd.read_csv(meta_csv_path)
        df_cells = pd.read_csv(cell_meta_csv_path)

        # What column header are image IDs stored under?
        if cell_or_image == 'cell':
            colheader = 'Image'
        else:
            colheader = 'ID'

        # Load and mix ensemble
        probs = None
        for predfile, mask in zip(BF_PREDICTIONS, MASKS):
            filename = os.path.join(predfile, r'prob_' + DATASET + '.npy')
            feature_fname = os.path.join(predfile, r'feat_' + DATASET + '.npy')

            # Parse the probability and feature files
            raw_probs = np.load(filename, allow_pickle=True)
            prob_img_list = raw_probs[:, 0]
            load_probs = raw_probs[:, 1:(len(LBL_NAMES) + 1)]

            load_probs = np.multiply(load_probs, mask)

            if probs is None:
                probs = load_probs
            else:
                probs += load_probs
        rev_probs = probs

        # Batch up the dataset
        num_batches = len(df_test) // BATCH_SIZE
        partial_batch = False
        if len(df_test) % BATCH_SIZE > 0:
            num_batches += 1
            partial_batch = True

        batch_end = 0

        if DEBUG:
            num_batches = 1

        X = []
        y = []
        list_cellprob = []

        if cell_or_image == 'cell':
            rev_prob_img = np.array([i.split('+')[0] for i in raw_probs[:, 0]])
            rev_cell_IDs = np.array([i.split('+')[1] for i in raw_probs[:, 0]])

        # For each batch in the dataset
        for batch in tqdm(range(0, num_batches)):

            cell_count_within_batch = 0

            batch_beg = batch_end
            batch_end = batch_beg + BATCH_SIZE
            if batch_end > len(df_test):
                batch_end = len(df_test)
            if batch_beg == batch_end:
                break

            df_batch = df_test[batch_beg:batch_end]

            img_list = df_batch[colheader].to_numpy()

            if cell_or_image == 'image':
                batch_probs = rev_probs[batch_beg:batch_end].astype(float)
                image_IDs = img_list
            else:
                #min_index = np.min(np.where(rev_prob_img == df_batch[colheader].to_numpy()[0]))
                #max_index = np.max(np.where(rev_prob_img == df_batch[colheader].to_numpy()[len(df_batch) - 1]))
                batch_probs = rev_probs[batch_beg:batch_end]
                image_IDs = rev_prob_img[batch_beg:batch_end]
                cell_IDs = rev_cell_IDs[batch_beg:batch_end]

            # Confirm that our features and probabilities match up to our img_list
            if cell_or_image == 'image':
                assert (np.array_equal(img_list, prob_img_list[batch_beg:batch_end]))

            img_data = []

            predictions = []
            pred_strings = []

            cellindex = 0

            count = 0  # Tracker for probability counting

            #counts = df_batch.groupby('ImageID').size()

            #for ID, mask, bbox in zip(img_list, masks, bboxes):
            for ID, ImageID in zip(img_list, image_IDs):

                if cell_or_image == 'image':
                    prob = batch_probs[count]
                    count += 1
                    cellcount = len(df_cells[df_cells['ImageID'] == ImageID])
                    cellIDs = df_cells[df_cells['ImageID'] == ImageID]['CellID'].tolist()
                else:
                    #cellcount_check = df_batch[df_batch[colheader] == ID].count()[0]
                    #cellcount = cnt
                    cellcount = 1

                probset = np.zeros(shape=(cellcount, len(LBL_NAMES) + 2), dtype='object')  # was len(prob) + 2
                cellprob = np.zeros(shape=(cellcount, len(LBL_NAMES)), dtype='float32')  # was len(prob)

                if truncate != len(LBL_NAMES):
                    indices = np.argpartition(prob[0:len(LBL_NAMES) - 1], -truncate)[0:len(LBL_NAMES) - truncate]
                    prob[indices] = 0.0
                else:
                    indices = np.ones(len(LBL_NAMES), dtype=bool)

                if cell_or_image == 'image':
                    for it in range(0, len(probset)):
                        cellprob[it] = prob
                        probset[it, 0] = ImageID  # Image ID
                        probset[it, 1] = cellIDs[it]  # Cell ID
                        probset[it, 2:(2 + len(LBL_NAMES))] = np.expand_dims(prob.copy(), axis=0)
                    list_cellprob.append(cellprob)
                else:
                    probset[:, 0] = ImageID  # Image ID
                    probset[:, 1] = df_batch[df_batch[colheader] == ID]['CellID']
                    # = np.arange(1, len(probset) + 1)  # Cell ID
                    probset[:, 2:(2 + len(LBL_NAMES))] = batch_probs[
                                                         cell_count_within_batch:cell_count_within_batch + len(
                                                             probset)]
                    cell_count_within_batch += len(probset)
                    list_cellprob.append([])

                assert (probset[0][0] == probset[len(probset) - 1][0])  # Make sure probset is all for the same cell

                # Keep track of all predictions to save to a pre-ensemble file
                if 'full_probset' not in locals():
                    full_probset = probset
                else:
                    full_probset = np.concatenate((full_probset, probset))

        aug_probset.append(full_probset)
        del full_probset

    # Rank the test augmentations
    if RANK:
        for entry in aug_probset:
            preds = entry[:, 2:]
            entry[:, 2:] = rankdata(a=preds, method='average', axis=0) / len(preds)

    # Average the test augmentations
    if len(aug_probset) > 1:
        aggregate = np.zeros((aug_probset[0].shape[0], aug_probset[0].shape[1] - 2), dtype=float)
        for entry in aug_probset:
            aggregate = np.add(aggregate, entry[:, 2:])
        aggregate = np.divide(aggregate, len(aug_probset))
        full_probset = np.hstack((aug_probset[0][:, 0:2], aggregate))
    else:
        full_probset = aug_probset[0]

    # Save out intermediate probability files
    if args.out_path == '':
        prob_fname = r'F:\probabilities_' + args.model + '_' + args.epoch + '_' + datetime.now().strftime(
            "%Y%m%d-%H%M%S") + '.csv'
    else:
        if args.add_datetime > 0:
            prob_fname = os.path.join(args.out_path,
                                      'probabilities_' + args.model + '_' + args.epoch + '_' + datetime.now().strftime(
                                          "%Y%m%d-%H%M%S") + '.csv')
        else:
            prob_fname = os.path.join(args.out_path,
                                      'probabilities_' + args.model + '_' + args.epoch + '.csv')
    columns = ['ImageID', 'CellID']
    columns.extend(LBL_NAMES)
    df_probs = pd.DataFrame(data=full_probset, columns=columns)
    df_probs.to_csv(prob_fname, index=False)

if __name__ == '__main__':
    reformat_preds()
    print("Done!")

