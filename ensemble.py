from scipy.stats import rankdata
import argparse
import ntpath

from HPAutils import *

# Arguments
parser = argparse.ArgumentParser(description='PyTorch Protein Classification')
parser.add_argument('--ensemble', default='', type=str, help='ensemble configuration file name')
parser.add_argument('--dataset', default='test')
parser.add_argument('--submission', default='1')
parser.add_argument('--rank', default=1, type=int, help='use rank-based ensembling')
parser.add_argument('--meta_csv', default='')
parser.add_argument('--cell_meta_csv_path', default='')
parser.add_argument('--out_path', default='')
parser.add_argument('--batch_size', default=25)
parser.add_argument('--add_datetime', default=1)


class Ensemble_Arguments():
    def __init__(self):
        self.ensemble = ''
        self.dataset = 'test'
        self.submission = '1'
        self.rank = 1
        self.meta_csv = ''
        self.cell_meta_csv = ''
        self.out_path = ''
        self.batch_size = 25
        self.add_datetime = 1

def ensemble(args=None):
    if args == None:
        args = parser.parse_args()

    ensemble_fname = args.ensemble
    ensemble_df = pd.read_csv(ensemble_fname)

    BATCH_SIZE = int(args.batch_size)

    # Parse Arguments
    DATASET = args.dataset
    if args.rank == 1:
        RANK = True
    else:
        RANK = False
    if args.submission == '1':
        SUBMISSION = True
    else:
        SUBMISSION = False

    if args.cell_meta_csv_path == '':
        df_cell_meta = pd.read_csv('F:\\public_cell_sample_submission.csv')
    else:
        df_cell_meta = pd.read_csv(args.cell_meta_csv_path)

    if args.meta_csv == '':
        df_out = pd.read_csv('X:\\' + DATASET + '_sample_submission.csv')
    else:
        df_out = pd.read_csv(args.meta_csv)

    # Get the names of files we want to ensemble together
    BF_PREDICTIONS = ensemble_df['Predictions'].values
    TYPES = ensemble_df["Type"].values
    WEIGHTS = ensemble_df[ensemble_df.columns[2:]].values
    WEIGHTS = np.divide(WEIGHTS, np.sum(WEIGHTS, axis=0))

    # Combine predictions into a single merged set
    for predfile, type, weight in zip(BF_PREDICTIONS, TYPES, WEIGHTS):
        if type == 'cell':
            df_test = pd.read_csv(predfile)
            images = df_test[['ImageID', 'CellID']].to_numpy()
            preds = df_test[LBL_NAMES].astype('float').to_numpy()
            all = df_test.to_numpy()

            if RANK:
                preds = rankdata(a=preds, method='average', axis=0) / len(preds)

            preds = np.multiply(preds, weight)

            # Weight the predictions
            if 'merged' in locals():
                merged += preds
            else:
                merged = preds
        if type == 'image':
            df_test = pd.read_csv(predfile)
            preds = df_test[LBL_NAMES].astype('float').to_numpy()

            if RANK:
                preds = rankdata(a=preds, method='average', axis=0) / len(preds)

            preds = np.multiply(preds, weight)

            # Weight the predictions
            if 'merged_image' in locals():
                merged_image += preds
            else:
                merged_image = preds

    # Combine image and cell level together according to heuristic
    merged[:, 0:18] = np.multiply(merged[:, 0:18], merged_image[:, 0:18])
    merged[:, 18] = np.maximum(merged[:, 18], merged_image[:, 18])
    #merged[:, 18] = np.mean([merged[:, 18], merged_image[:, 18]], axis=0)

    # Get distinct list of image IDs
    img_list = df_test['ImageID'].unique()

    num_batches = len(img_list) // BATCH_SIZE
    partial_batch = False

    if len(img_list) % BATCH_SIZE > 0:
        num_batches += 1
        partial_batch = True
    batch_end = 0

    for batch in tqdm(range(0, num_batches)):
        batch_beg = batch_end
        batch_end = batch_beg + BATCH_SIZE
        if batch_end > len(df_test):
            batch_end = len(df_test)
        if batch_beg == batch_end:
            break

        # Go through each image id
        for ID in img_list[batch_beg:batch_end]:

            # Extract the corresponding data from the merged probabilities
            ID_indices = np.where(all[:, 0] == ID)
            probset = merged[np.min(ID_indices):np.max(ID_indices) + 1]
            probset = neg_heuristic(probset)

            # New decay by image
            #probset = mitotic_decay(probset)
            encoded_strings = df_cell_meta.loc[df_cell_meta['ImageID'] == ID]['PredictionString']

            # Determine sets of high probability overlapping mitotic cells
            # For each set:
            #   Rebuild the mask
            #   Decode strings
            #   Combine masks together
            #   Re-encode strings
            #   Average probabilities

            #   Modify build prediction string so it ignores duplicate strings

            if SUBMISSION:
                # Construct prediction string
                pred_string = build_prediction_string_precoded(probset, encoded_strings)

                # Update dataframe
                df_out.loc[df_out['ID'] == ID, 'PredictionString'] = pred_string

    # Save output
    print("Submission generation: " + str(SUBMISSION))

    # Save out submission files
    ensemble_name = ntpath.basename(args.ensemble).split('.')[0]
    if SUBMISSION:
        if args.out_path == '':
            fname = r'F:\submission_' + ensemble_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
        else:
            if args.add_datetime > 0:
                fname = os.path.join(args.out_path, 'submission_' + ensemble_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv')
            else:
                fname = os.path.join(args.out_path, 'submission_' + ensemble_name + '.csv')
        print(fname)
        df_out.to_csv(fname, index=False)

    # Save out intermediate probability files
    if args.out_path == '':
        prob_fname = r'F:\probabilities_' + ensemble_name + '_' + datetime.now().strftime(
            "%Y%m%d-%H%M%S") + '.csv'
    else:
        if args.add_datetime > 0:
            prob_fname = os.path.join(args.out_path, 'probabilities_' + ensemble_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv')
        else:
            prob_fname = os.path.join(args.out_path, 'probabilities_' + ensemble_name + '.csv')
    columns = ['ImageID', 'CellID']
    columns.extend(LBL_NAMES)
    out = np.hstack((images, merged))
    df_probs = pd.DataFrame(data=out, columns=columns)
    df_probs.to_csv(prob_fname, index=False)

def mitotic_decay(probset):
    mit_unique = np.flip(np.unique(probset[:, 11]))
    ranks = np.sqrt(np.arange(len(mit_unique)) + 1)
    dict = {A: B for A, B in zip(mit_unique, ranks)}
    for i in range(0, len(probset)):
        probset[i, 11] /= dict[probset[i, 11]]
    return probset

def neg_heuristic(probset): # Reduce probablities uniformly by probability that it is a negative cell
    probset[:, 0:18] = np.multiply(probset[:, 0:18], (1 - probset[:, 18]).reshape(-1, 1))
    #if probset[:, 18] > 0.07:
    #    q = min(1, -math.log10(probset[:, 18]*10))
    #    probset[:, 0:18] = np.maximum(0.0, np.multiply(probset[:, 0:18], q))
    return probset

if __name__ == '__main__':
    ensemble()
    print("Done!")



