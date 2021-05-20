import torch
import argparse
from torch.multiprocessing import Pool
from HPAutils import *
from hpacellsegandrew import cellsegmentator



def load_image_from_numpy(path, image_file, drop_yellow=False, channel_mix=False):
    filename = os.path.join(path, image_file)
    assert os.path.exists(filename), f'not found {filename}'
    img = np.load(os.path.join(path, image_file))

    if drop_yellow:
        img = img[:, :, 0:3]

    if channel_mix:
        green = img[:, :, 1]
        yellow = img[:, :, 3]
        mixed = np.maximum(green, yellow)
        img[:, :, 1] = mixed
    return img

def recombine_meta(META_OUT):
    files = os.listdir(META_OUT)

    file = files[0]
    meta_df = pd.read_csv(os.path.join(META_OUT, file))
    os.remove(os.path.join(META_OUT, file))

    if len(files) > 0:
        for file in files[1:]:
            data = pd.read_csv(os.path.join(META_OUT, file))
            os.remove(os.path.join(META_OUT, file))
            meta_df = meta_df.append(data, ignore_index=True)

    meta_df['CellID'] = pd.to_numeric(meta_df['CellID'])
    meta_df.sort_values(by=['Image', 'CellID'], inplace=True, ignore_index=True)
    meta_df.to_csv(os.path.join(META_OUT, 'sample_submission_cell.csv'), index=False)


def _do_raw_segmentation(arg):
    df_test = arg['df_test']
    BATCH_SIZE = arg['batch_size']
    batch_start = arg['batch_start']
    batch_count = arg['batch_count']
    ROOT = arg['ROOT']
    NUC_MODEL = arg['NUC_MODEL']
    CELL_MODEL = arg['CELL_MODEL']
    CACHE_PATH = arg['CACHE_PATH']
    NUCLEI_ONLY = arg['NUCLEI_ONLY']

    pid = os.getpid()

    print('Process: ' + str(pid))
    print('CUDA enabled' + str(torch.cuda.is_available()))

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    segmentator = cellsegmentator.CellSegmentator(
        NUC_MODEL,
        CELL_MODEL,
        scale_factor=0.25,
        device=device,
        padding=True,
        multi_channel_model=True
    )

    num_batches = batch_start + batch_count

    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)
    existing_cache = os.listdir(CACHE_PATH)

    # Segmentation loop
    batch_end = batch_start * BATCH_SIZE  # Pretend we've already covered batches prior to this
    for batch in range(batch_start, num_batches):
        torch.cuda.empty_cache()

        batch_beg = batch_end
        batch_end = batch_beg + BATCH_SIZE
        if batch_end > len(df_test):
            batch_end = len(df_test)
        if batch_beg == batch_end:
            break

        print("Segment batch starting at: " + str(batch_beg) + " batch " + str(batch) + " out of " + str(num_batches))

        # Get the next batch
        df_batch = df_test[batch_beg:batch_end]

        if 'ID' in df_batch.columns:
            header = 'ID'
        else:
            header = 'Image'

        img_list = df_batch[header].tolist()
        img_data = []

        # Remove files we already have data for
        to_remove = []
        for ID in img_list:
            fname = ID + '_nuc.npy'
            if fname in existing_cache:
                to_remove.append(ID)

        for removal in to_remove:
            img_list.remove(removal)

        img_list = np.array(img_list)

        for ID in img_list:
            if os.path.exists(os.path.join(r'X:\ext\npy.full.rgby', ID + '.npy')):
                img_data.append(load_image_from_numpy(r'X:\ext\npy.full.rgby', ID + '.npy'))
            else:
                img_data.append(load_RGBY_image(ROOT, ID))

        # print(img_list)
        write_raw_segments(segmentator, img_data, img_list, CACHE_PATH, 4, NUCLEI_ONLY)
        torch.cuda.empty_cache()
    del segmentator

        # print(os.listdir(CACHE_PATH))


def _segment_and_crop_batches(arg):
    df_test = arg['df_test']
    BATCH_SIZE = arg['batch_size']
    batch_start = arg['batch_start']
    batch_count = arg['batch_count']
    ROOT = arg['ROOT']
    CROP_OUT = arg['CROP_OUT']
    IMG_OUT = arg['IMG_OUT']
    EXTRA_CROPS = arg['EXTRA_CROPS']
    EXTRA_IMGS = arg['EXTRA_IMGS']
    META_OUT = arg['META_OUT']
    CROP_SIZE = arg['CROP_SIZE']
    CACHE_PATH = arg['CACHE_PATH']
    NUCLEI_ONLY = arg['NUCLEI_ONLY']
    DONT_WRITE_ORIGINAL = arg['DONT_WRITE_ORIGINAL']

    pid = os.getpid()

    num_batches = batch_start + batch_count

    out_images = []
    out_imgids = []
    out_cellids = []
    out_sizes = []
    out_prediction_strings = []

    # Segmentation loop
    batch_end = batch_start * BATCH_SIZE  # Pretend we've already covered batches prior to this
    for batch in range(batch_start, num_batches):

        batch_beg = batch_end
        batch_end = batch_beg + BATCH_SIZE
        if batch_end > len(df_test):
            batch_end = len(df_test)
        if batch_beg == batch_end:
            break

        print("Segment batch starting at: " + str(batch_beg) + " batch " + str(batch) + " out of " + str(num_batches))

        # Get the next batch
        df_batch = df_test[batch_beg:batch_end]

        if 'ID' in df_batch.columns:
            header = 'ID'
        else:
            header = 'Image'

        img_list = df_batch[header].to_numpy()
        img_data = []

        for ID in img_list:
            if os.path.exists(os.path.join(r'X:\ext\npy.full.rgby', ID + '.npy')):
                image = load_image_from_numpy(r'X:\ext\npy.full.rgby', ID + '.npy')
                img_data.append(image)
            else:
                image = load_RGBY_image(ROOT, ID)
                img_data.append(image)
            for extraimg in EXTRA_IMGS:
                if not os.path.exists(os.path.join(IMG_OUT, str(extraimg))):
                    os.makedirs(os.path.join(IMG_OUT, str(extraimg)))
                path = os.path.join(IMG_OUT, str(extraimg), ID + '.npy')
                resized = cv2.resize(image, (int(extraimg), int(extraimg)), interpolation=cv2.INTER_LINEAR)
                np.save(path, resized, allow_pickle=True)

        # if img_list[0] != '020a29cf-2c24-478b-8603-c22a90dc3e31':
        #    continue

        # masks, nuc_masks = get_masks(segmentator, img_data)
        masks, nuc_masks = label_raw_segments(img_list, CACHE_PATH, NUCLEI_ONLY, scale_factor=0.25)
        if NUCLEI_ONLY:
            bboxes = get_bboxes(nuc_masks)
            masks = nuc_masks
        else:
            bboxes = get_bboxes(masks)

        for mask, nuc_mask, ID, image, bboxset in zip(masks, nuc_masks, img_list, img_data, bboxes):
            # if ID != '020a29cf-2c24-478b-8603-c22a90dc3e31':
            #    continue

            # Cull cells and get the prediction strings
            # try:
            if not NUCLEI_ONLY:
                pred_strs = cull_and_string(mask, True, nuc_mask)
            else:
                pred_strs = np.empty((len(bboxset)), dtype='str')
            # except:
            #    print(ID + ' Failed')
            for idx, bbox, pred_str in zip(range(1, len(bboxset) + 1), bboxset, pred_strs):
                if pred_str == 'Invalid':
                    continue  # Skip culled cells
                else:
                    # Get and write image crop
                    data = crop_mask_resize_img(image, idx, mask, bbox, 4, CROP_SIZE)
                    if NUCLEI_ONLY:
                        fname = os.path.join(CROP_OUT, ID + '_nuc+' + str(idx) + '.npy')
                    else:
                        fname = os.path.join(CROP_OUT, ID + '+' + str(idx) + '.npy')
                    if not DONT_WRITE_ORIGINAL:
                        np.save(fname, data, allow_pickle=True)

                    for extracrop in EXTRA_CROPS:
                        scaled = cv2.resize(data, (int(extracrop), int(extracrop)), interpolation=cv2.INTER_LINEAR)
                        if not os.path.exists(os.path.join(CROP_OUT, str(extracrop))):
                            os.makedirs(os.path.join(CROP_OUT, str(extracrop)))
                        if NUCLEI_ONLY:
                            fname = os.path.join(CROP_OUT, str(extracrop), ID + '_nuc+' + str(idx) + '.npy')
                        else:
                            fname = os.path.join(CROP_OUT, str(extracrop), ID + '+' + str(idx) + '.npy')
                        np.save(fname, scaled, allow_pickle=True)

                    # Prep metadata
                    out_images.append(ID)
                    if NUCLEI_ONLY:
                        out_imgids.append(str(ID) + '_nuc+' + str(idx))
                    else:
                        out_imgids.append(str(ID) + '+' + str(idx))
                    out_cellids.append(int(idx))
                    out_sizes.append(int(mask.shape[0]))
                    out_prediction_strings.append(pred_str)

    # Write metadata for this process out after all batches are complete
    out_images = np.stack(out_images)
    out_imgids = np.stack(out_imgids)
    out_cellids = np.stack(out_cellids)
    out_sizes = np.stack(out_sizes)
    out_prediction_strings = np.stack(out_prediction_strings)

    meta = np.vstack((out_imgids, out_images, out_cellids, out_sizes, out_sizes, out_prediction_strings)).T
    dataset = pd.DataFrame(meta,
                           columns=['Image', 'ImageID', 'CellID', 'ImageWidth', 'ImageHeight', 'PredictionString'])
    fname = os.path.join(META_OUT, 'cell_meta_' + str(pid) + '.csv')
    print("Writing " + fname)
    dataset.to_csv(fname, index=False)

def private_segment(df_test, ROOT, META_OUT, CROP_OUT, IMG_OUT, CACHE_OUT, NUC_MODEL, CELL_MODEL, BATCH_SIZE=4, workers=0, EXTRA_CROPS=[], EXTRA_IMGS=[], NUCLEI_ONLY=False, DONT_WRITE_ORIGINAL=False):
    # Batch up the dataset
    if __name__ == '__main__':

        # Batch up the dataset
        # mp.set_start_method('spawn') # Use spawn method

        num_batches = len(df_test) // BATCH_SIZE

        if workers is None:
            n_cpu = min(4, os.cpu_count())
            print('n_cpu: ', n_cpu)
        else:
            n_cpu = workers

        if len(df_test) % BATCH_SIZE > 0:  # Determine if we have a partial batch to deal with
            num_batches += 1

        ######## RESUME CODE################
        #skip_batches = 2457
        skip_batches = 0
        ###################################
        num_batches -= skip_batches

        batches_per_process = num_batches // n_cpu

        # Chunk up the data
        args = []

        for i in range(0, n_cpu):
            if i == n_cpu - 1:
                extrabatches = num_batches % n_cpu
            else:
                extrabatches = 0

            args.append({
            'df_test': df_test,
            'batch_size': BATCH_SIZE,
            'ROOT': ROOT,
            'batch_start': skip_batches + i * batches_per_process,
            'batch_count': batches_per_process + extrabatches,
            'CROP_OUT': CROP_OUT,
            'META_OUT': META_OUT,
            'IMG_OUT': IMG_OUT,
            'CROP_SIZE': 768,
            'NUC_MODEL': NUC_MODEL,
            'CELL_MODEL': CELL_MODEL,
            'CACHE_PATH': CACHE_OUT,
            'EXTRA_CROPS': EXTRA_CROPS,
            'EXTRA_IMGS': EXTRA_IMGS,
            'NUCLEI_ONLY': NUCLEI_ONLY,
            'DONT_WRITE_ORIGINAL': DONT_WRITE_ORIGINAL
            })

        newargs = args[0].copy()
        newargs['batch_count'] = num_batches
        newargs['batch_start'] = 0
        _do_raw_segmentation(newargs)

        if workers > 1:
            p = Pool(processes=n_cpu)
            p.map(_segment_and_crop_batches, args)

            print(f"multi processing complete.")

            p.close()
            p.join()
        else:
            print("Single processor")
            _segment_and_crop_batches(args[0])

        # recombine metadata for both
        recombine_meta(META_OUT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform segmentation')

    parser.add_argument('--metadata', default=r'D:\HPA\train.csv', type=str)
    parser.add_argument('--root_dir', default='D:\\HPA\\train', type=str)
    parser.add_argument('--nuc_model', default=r'D:\HPA\input\hpacellsegmentatormodelweights\dpn_unet_nuclei_v1.pth', type=str)
    parser.add_argument('--cell_model', default=r'D:\HPA\input\hpacellsegmentatormodelweights\dpn_unet_cell_3ch_v1.pth', type=str)
    parser.add_argument('--meta_dir', default=r'D:\Metadata', type=str)
    parser.add_argument('--crop_dir', default=r'D:\Crops', type=str)
    parser.add_argument('--img_dir', default=r'D:\Imgs')
    parser.add_argument('--cache_dir', default=r'D:\ScaledTrainCache', type=str)

    args = parser.parse_args()

    df_test = pd.read_csv(args.metadata)

    for dir in [args.meta_dir, args.crop_dir, args.img_dir, args.cache_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Remove
    df_test = df_test.head(1)

    ROOT = args.root_dir

    NUC_MODEL = args.nuc_model
    CELL_MODEL = args.cell_model

    private_segment(df_test, ROOT, args.meta_dir, args.crop_dir, args.img_dir, CACHE_OUT=args.cache_dir, NUC_MODEL=NUC_MODEL, CELL_MODEL=CELL_MODEL, BATCH_SIZE=1, workers=1, EXTRA_CROPS=[], EXTRA_IMGS=[768], NUCLEI_ONLY=False, DONT_WRITE_ORIGINAL=False)
    print("Done!")