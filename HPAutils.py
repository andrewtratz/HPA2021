import base64
import os
import numpy as np
import pandas as pd
from pycocotools import _mask as coco_mask
import pycocotools
import typing as t
import zlib
import cv2
from datetime import datetime
from tqdm import tqdm
# from hpacellseg.utils import label_cell
from hpacellsegandrew.utils import label_cell, label_nuclei
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import numpy.ma as ma

from cv2 import CV_8U

import mlcrate as mlc

IMG_PATHS = {
    'test': r'F:\test\npy.512.rgby',
    'custom': r'X:\TestFiles\test\images_768',
    'custom512': r'F:\TestFiles512\test\images_512'
}

LBL_NAMES = ["Nucleoplasm", "Nuclear Membrane", "Nucleoli", "Nucleoli Fibrillar Center", "Nuclear Speckles",
             "Nuclear Bodies", "Endoplasmic Reticulum", "Golgi Apparatus", "Intermediate Filaments", "Actin Filaments",
             "Microtubules", "Mitotic Spindle", "Centrosome", "Plasma Membrane", "Mitochondria", "Aggresome", "Cytosol",
             "Vesicles", "Negative"]


def resize_mask(mask, newsize):
    newmask = cv2.resize(src=mask, dsize=(newsize[1], newsize[0]), interpolation=cv2.INTER_NEAREST)
    return newmask


def load_mask_and_labels(path, image_id):
    maskname = 'm=' + image_id + '.png'
    labelname = 'zzlbl=' + image_id + '.csv'

    path_mask = os.path.join(path, maskname)
    path_labels = os.path.join(path, labelname)

    mask = cv2.imread(path_mask, flags=cv2.IMREAD_UNCHANGED)
    labels = pd.read_csv(path_labels).to_numpy()[:, 0:len(LBL_NAMES)]

    return mask, labels


def load_images(path, label_id, image):
    normal = image
    ych = 'y=' + image
    mask = 'm=' + image

    labelfolder = LBL_NAMES[label_id]

    path_normal = os.path.join(path, labelfolder, normal)
    path_ych = os.path.join(path, labelfolder, ych)
    path_mask = os.path.join(path, labelfolder, mask)

    img = cv2.imread(path_normal, flags=cv2.IMREAD_UNCHANGED)
    ychannel = cv2.imread(path_ych, flags=cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(path_mask, flags=cv2.IMREAD_UNCHANGED)

    return img, ychannel, mask


def visualize(path, img_id, mask=None):
    img = load_RGBY_image(path, img_id)

    # Plotting Segmentations
    f, ax = plt.subplots(1, 2, figsize=(16, 16))
    ax[0].imshow(img)
    ax[0].set_title('Original Cells', size=20)
    if mask is not None:
        ax[1].imshow(mask)
        ax[1].set_title('Segmented Cells', size=20)
    plt.show()


def rle_to_mask(rle_string, height, width):
    """ Convert RLE sttring into a binary mask

    Args:
        rle_string (rle_string): Run length encoding containing
            segmentation mask information
        height (int): Height of the original image the map comes from
        width (int): Width of the original image the map comes from

    Returns:
        Numpy array of the binary segmentation mask for a given cell
    """
    rows, cols = height, width
    rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
    rle_pairs = np.array(rle_numbers).reshape(-1, 2)
    img = np.zeros(rows * cols, dtype=np.uint8)
    for index, length in rle_pairs:
        index -= 1
        img[index:index + length] = 255
    img = img.reshape(cols, rows)
    img = img.T
    return img


def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s" %
            mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" %
            mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode()

def write_raw_segments(segmentator, imgs, img_ids, cache_path, channels=4, nuc_only=False):
    # try:
    if nuc_only:
        nuc_images = [img[:,:,2] for img in imgs]
    else:
        if channels == 4:
            images = [np.concatenate(
                (np.expand_dims(img[:, :, 0], 2), np.expand_dims(img[:, :, 3], 2), np.expand_dims(img[:, :, 2], 2)), axis=2)
                      for img in imgs]
            nuc_images = [img[:, :, 2] for img in imgs]
            # images = [[img[:, :, 0] for img in imgs], # Microtubules
            #          [img[:, :, 3] for img in imgs], # ER
            #          [img[:, :, 2] for img in imgs]] # Nuclei
        elif channels == 3:
            images = imgs
            nuc_images = [img[:, :, 2] for img in imgs]
            # images = [[img[:, :, 0] for img in imgs],  # Microtubules
            #          [img[:, :, 1] for img in imgs],  # ER/Green
            #          [img[:, :, 2] for img in imgs]]  # Nuclei

    print("Segmenting...")
    nuc_segmentations = segmentator.pred_nuclei(nuc_images)

    if nuc_only:
        for nuc_seg, img_id in zip(nuc_segmentations, img_ids):
            nuc_path = os.path.join(cache_path, img_id + '_nuc.npy')
            np.save(nuc_path, nuc_seg, allow_pickle=True)
    else:
        # print(datetime.now())
        cell_segmentations = segmentator.pred_cells(images, precombined=True)

        for nuc_seg, cell_seg, img_id in zip(nuc_segmentations, cell_segmentations, img_ids):
            nuc_path = os.path.join(cache_path, img_id + '_nuc.npy')
            np.save(nuc_path, nuc_seg, allow_pickle=True)
            cell_path = os.path.join(cache_path, img_id + '_cell.npy')
            np.save(cell_path, cell_seg, allow_pickle=True)

def label_raw_segments(img_ids, cache_path, nuc_only=False, scale_factor=1.0):
    cell_segmentations = []
    nuc_segmentations = []
    cell_masks = []
    nuc_masks = []

    # Load the cached raw segment data
    for img_id in img_ids:
        nuc_path = os.path.join(cache_path, img_id + '_nuc.npy')
        cell_path = os.path.join(cache_path, img_id + '_cell.npy')
        nuc_segmentations.append(np.load(nuc_path, allow_pickle=True))
        if not nuc_only:
            cell_segmentations.append(np.load(cell_path, allow_pickle=True))

    pid = os.getpid()

    if nuc_only:
        for i in tqdm(range(len(nuc_segmentations)), desc=str(pid) + ': Labeling nuclei..'):
            nuc_mask = label_nuclei(nuc_segmentations[i], scale_factor=scale_factor)
            nuc_masks.append(nuc_mask)
    else:
        for i in tqdm(range(len(cell_segmentations)), desc=str(pid) + ': Labeling cells..'):
            nuc_mask, cell_mask = label_cell(nuc_segmentations[i], cell_segmentations[i], scale_factor=scale_factor)
            cell_masks.append(cell_mask)
            nuc_masks.append(nuc_mask)
    print(datetime.now())
    return cell_masks, nuc_masks

# Based on full image cell mask, locate bounding boxes for each individual cell
# Store in array of array of tuples
def get_bboxes(cell_masks):
    bboxes = []
    for cell_mask in cell_masks:
        regions = regionprops(cell_mask)

        bbox = []
        for props in regions:
            min_row, min_col, max_row, max_col = props.bbox
            bbox.append((min_row, min_col, max_row, max_col))
        bboxes.append(bbox)

    return bboxes


# image loader
def load_RGBY_image(path, image_id, image_size=None, pad=True):
    red = read_img(path, image_id, "red", image_size, pad)
    green = read_img(path, image_id, "green", image_size, pad)
    blue = read_img(path, image_id, "blue", image_size, pad)
    yellow = read_img(path, image_id, "yellow", image_size, pad)
    stacked_images = np.transpose(np.array([red, green, blue, yellow]), (1, 2, 0))  # Put channel in last dimension
    return stacked_images


# DEBUG
def read_img(path, image_id, color, image_size=None, pad=True):
    filename = os.path.join(path, image_id + '_' + color + '.png')
    assert os.path.exists(filename), f'not found {filename}'
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    #img = np.array(img, dtype='uint8') # Seems broken
    #img = cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))
    if np.max(img) > 255:
        img = (img / 257).astype('uint8')

    old_size = (img.shape[1], img.shape[0])  # old_size is in (height, width) format

    if image_size is not None:
        ratio = float(image_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format

        # Only shrink if we're padding, never grow!
        if not (new_size[1] > old_size[1] and new_size[0] > old_size[1] and pad):
            img = cv2.resize(img, (new_size[1], new_size[0]))
        else:
            new_size = old_size

        if pad:
            delta_w = image_size - new_size[1]
            delta_h = image_size - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            color = 0
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                     value=color)
    return img


def scale_mask_make_alpha(cell_idx, cell_mask, bbox, image_size=None):
    (min_row, min_col, max_row, max_col) = bbox

    mask = cell_mask[min_row:max_row, min_col:max_col]
    mask = (mask != cell_idx).astype('uint8') * 255  # Make things not matching mask transparent per PNG specification

    if image_size is not None:
        old_size = (max_row - min_row + 1, max_col - min_col + 1)  # old_size is in (height, width) format

        ratio = float(image_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # im = cv2.resize(mask, (new_size[1], new_size[0]))
        im = resize_mask(mask, new_size)

        delta_w = image_size - new_size[1]
        delta_h = image_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        alpha_mask = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=0)
    else:
        alpha_mask = mask
    return alpha_mask


def crop_mask_resize_img(image, cell_idx, cell_mask, bbox, channel_depth=3, image_size=None):
    (min_row, min_col, max_row, max_col) = bbox

    mask = cell_mask[min_row:max_row, min_col:max_col]
    mask = (mask == cell_idx).astype('uint8')

    if channel_depth == 3:
        mask_3 = np.stack((mask, mask, mask), axis=-1)
        crop = image[min_row:max_row, min_col:max_col, :]
        if crop.shape != mask_3.shape:
            print("Uh oh!")
        crop = np.multiply(crop, mask_3)
    elif channel_depth == 4:
        mask_4 = np.stack((mask, mask, mask, mask), axis=-1)
        crop = image[min_row:max_row, min_col:max_col, :]
        if crop.shape != mask_4.shape:
            print("Uh oh!")
        crop = np.multiply(crop, mask_4)
    else:
        crop = np.multiply(image[min_row:max_row, min_col:max_col], mask)

    if image_size is not None:
        old_size = (max_row - min_row + 1, max_col - min_col + 1)  # old_size is in (height, width) format

        ratio = float(image_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format

        im = cv2.resize(crop, (new_size[1], new_size[0]))

        delta_w = image_size - new_size[1]
        delta_h = image_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        if channel_depth == 3:
            color = [0, 0, 0]
        elif channel_depth == 4:
            color = [0, 0, 0, 0]
        else:
            color = 0
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
    else:
        new_im = crop
    return new_im


def edges_to_remove(cell_mask, thresholds=[0.5]):
    out = []
    edges = []
    indices = []
    area = []
    dim = cell_mask.shape[0]
    # bboxes = get_bboxes([cell_mask])[0]

    cellmax = np.max(cell_mask)

    # for idx, box in zip(range(0, len(bboxes)), bboxes):
    for idx in range(0, cellmax):
        indices.append(idx + 1)
        # area.append((box[3]-box[1])*(box[2]-box[0]))
        area.append(np.count_nonzero(cell_mask == idx + 1))
        # if min(box) == 0 or max(box) == dim:
        if np.any(cell_mask[0, :] == idx + 1) or np.any(cell_mask[:, 0] == idx + 1) or np.any(
                cell_mask[-1, :] == idx + 1) or np.any(cell_mask[:, -1] == idx + 1):
            edges.append(True)
        else:
            edges.append(False)

    area = np.stack(area)
    edges = np.stack(edges)

    if np.any(~edges):
        avg_area = np.average(area[~edges])
    else:
        avg_area = np.average(area)

    output = []

    for threshold in thresholds:
        indices = np.stack(indices)
        low_area = (area < threshold * avg_area)
        bool_mask = np.logical_and(edges, low_area)

        micro_area = (area < 0.1 * avg_area)
        bool_mask = np.logical_or(bool_mask, micro_area)
        output.append(indices[bool_mask])

    return output, cellmax


def cull_indices(cell_mask, nuc_mask):
    nuc050_070, nuccount = edges_to_remove(nuc_mask, [0.5, 0.7])
    cell040 = edges_to_remove(cell_mask, [0.4])[0][0]
    nuc050 = nuc050_070[0]
    nuc070 = nuc050_070[1]
    indexes_to_remove = list(set(nuc050) | (set(cell040) & set(nuc070)))
    return indexes_to_remove, nuccount


# Cull non-conforming edge cells and return prediction strings and valid cell indices
def cull_and_string(cell_mask, edgeclean=True, nuc_mask=None):
    assert (nuc_mask is not None)

    if edgeclean:
        indexes_to_remove, cellcount = cull_indices(cell_mask, nuc_mask)
    else:
        indexes_to_remove, cellcount = [], np.max(cell_mask)

    pred_strs = []
    valid_cells = []

    for idx in range(0, cellcount):
        # Skip any cell we are removing
        if (idx + 1) in indexes_to_remove:
            pred_strs.append('Invalid')
            continue

        # Compute RLE encoded mask
        str_encoded = binary_mask_to_ascii(cell_mask, idx + 1)

        # Ignore totally blank masks
        if str_encoded in ['eNoLCAhIMgAABLkBgw==', 'eNoLCAgIMAEABJkBdQ==', 'eNoLCAgIsAQABJ4Beg==',
                           'eNoLCAjJNgIABNkBkg==', 'eNoLCAiwAAADDAEp']:
            pred_strs.append('Invalid')
            continue

        pred_strs.append(str_encoded)

    return pred_strs


# Build prediction string when we already have the encoded mask
def build_prediction_string_precoded(predictions, encoded_strings):
    pred_str = ''
    for idx, str_encoded in zip(range(0, len(predictions)), encoded_strings):
        # Build the prediction string for the image
        for label_ID, prediction in zip(range(0, len(LBL_NAMES)), predictions[idx]):
            if len(pred_str) > 0:
                pred_str += ' '
            pred_str += str(label_ID) + ' ' + str(prediction) + ' ' + str_encoded
    return pred_str


def build_prediction_string(cell_mask, predictions, condense=False, edgeclean=False, nuc_mask=None):
    if edgeclean:
        indexes_to_remove, _ = cull_indices(cell_mask, nuc_mask)
    else:
        indexes_to_remove = []

    pred_str = ''
    for idx in range(0, len(predictions)):

        # Skip any cell we are removing
        if (idx + 1) in indexes_to_remove:
            continue

        # Compute RLE encoded mask
        # str_encoded = encode_binary_mask((cell_mask == (idx+1)))
        str_encoded = binary_mask_to_ascii(cell_mask, idx + 1)

        # Ignore totally blank masks
        if str_encoded in ['eNoLCAhIMgAABLkBgw==', 'eNoLCAgIMAEABJkBdQ==', 'eNoLCAgIsAQABJ4Beg==',
                           'eNoLCAjJNgIABNkBkg==', 'eNoLCAiwAAADDAEp']:
            continue
        # Make RLE friendly string format
        # str_encoded = encoded_mask.decode('utf-8')

        # Build the prediction string for the image
        for label_ID, prediction in zip(range(0, len(LBL_NAMES)), predictions[idx]):
            if condense and prediction == 0.0:
                continue
            if len(pred_str) > 0:
                pred_str += ' '
            pred_str += str(label_ID) + ' ' + str(prediction) + ' ' + str_encoded

    return pred_str


def binary_mask_to_ascii(mask, mask_val=1):
    """Converts a binary mask into OID challenge encoding ascii text."""

    # NEEDS OPTIMIZATION ################### BOTH WHERE AND ASTYPE ARE SLOW
    # mask = np.where(mask == mask_val, 1, 0).astype(np.bool)
    # ones = np.ones_like(mask, dtype='uint8', order='F')
    # zeros = np.zeros_like(mask, dtype='uint8', order='F')
    mask = (mask == mask_val).astype('uint8')

    # check input mask --
    # if mask.dtype != np.bool:
    #  raise ValueError(f"encode_binary_mask expects a binary mask, received dtype == {mask.dtype}")

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(f"encode_binary_mask expects a 2d mask, received shape == {mask.shape}")

    # convert input mask to expected COCO API input --
    mask_to_encode = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

    if mask_to_encode.dtype != np.uint8:
        mask_to_encode = mask_to_encode.astype(np.uint8)

    if ~mask_to_encode.flags.f_contiguous:  # Make sure we are in Fortran order
        mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode()


def ascii_to_binary_mask(encodedstring, h, w):
    base64_str = base64.b64decode(encodedstring)
    binary_str = zlib.decompress(base64_str)

    # decoded_mask = coco_mask.decode(list(binary_str))
    # compressed_rle = coco_mask.frPyObjects(binary_str, binary_str.get('size')[0], binary_str.get('size')[1])
    RLE_obj = [{'size': [h, w], 'counts': binary_str}]
    decoded_mask = coco_mask.decode(RLE_obj)

    decoded_mask = decoded_mask.reshape((h, w))

    return decoded_mask


def rle_encoding(img, mask_val=1):
    """
  Turns our masks into RLE encoding to easily store them
  and feed them into models later on
  https://en.wikipedia.org/wiki/Run-length_encoding

  Args:
      img (np.array): Segmentation array
      mask_val (int): Which value to use to create the RLE

  Returns:
      RLE string

  """
    dots = np.where(img.T.flatten() == mask_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b

    return ' '.join([str(x) for x in run_lengths])


def rle_to_mask(rle_string, height, width):
    """ Convert RLE sttring into a binary mask

  Args:
      rle_string (rle_string): Run length encoding containing
          segmentation mask information
      height (int): Height of the original image the map comes from
      width (int): Width of the original image the map comes from

  Returns:
      Numpy array of the binary segmentation mask for a given cell
  """
    rows, cols = height, width
    rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
    rle_pairs = np.array(rle_numbers).reshape(-1, 2)
    img = np.zeros(rows * cols, dtype=np.uint8)
    for index, length in rle_pairs:
        index -= 1
        img[index:index + length] = 255
    img = img.reshape(cols, rows)
    img = img.T
    return img


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width]. Mask pixels are either 1 or 0.
    Returns: bbox array [(y1, x1, y2, x2)].
    """
    boxes = np.zeros([4], dtype=np.int32)
    m = mask
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]

    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0

    return x1, x2, y1, y2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Mapping function from first to second competition
def BFmap(probs):
    rev_probs = np.zeros((len(probs), 18))
    # Nuclear items
    rev_probs[:, 0:8] = probs[:, 0:8]
    # Intermediate filiments
    rev_probs[:, 8] = probs[:, 11]
    # Actin filiments
    rev_probs[:, 9] = probs[:, 12] + probs[:, 13] * (1 - probs[:, 12])
    # Microtubles
    rev_probs[:, 10] = probs[:, 14]
    # Mitotic spindle
    rev_probs[:, 11] = probs[:, 17]
    # Centrosome
    rev_probs[:, 12] = probs[:, 19]
    # Plasma membrane
    rev_probs[:, 13] = probs[:, 21] + probs[:, 22] * (1 - probs[:, 21])
    # Mitochondria
    rev_probs[:, 14] = probs[:, 23]
    # Aggresome
    rev_probs[:, 15] = probs[:, 24]
    # Cytosol
    rev_probs[:, 16] = probs[:, 25]
    # Vesicles
    rev_probs[:, 17] = probs[:, 8]
    rev_probs[:, 17] += probs[:, 9] * (1 - rev_probs[:, 17])
    rev_probs[:, 17] += probs[:, 10] * (1 - rev_probs[:, 17])
    rev_probs[:, 17] += probs[:, 20] * (1 - rev_probs[:, 17])
    rev_probs[:, 17] += probs[:, 26] * (1 - rev_probs[:, 17])

    return rev_probs


class MaskReducer:
    def __init__(self, size=512, dim=16):
        fact = size // dim
        self.fact = fact
        self.size = size
        self.dim = dim
        self.checkers = np.zeros((size, size, dim, dim), dtype=bool)  # Create large False array

        x_channel = 0
        for x_offset in range(0, dim):
            y_channel = 0
            for y_offset in range(0, dim):
                for x in range(x_offset * fact, (x_offset + 1) * fact):
                    for y in range(y_offset * fact, (y_offset + 1) * fact):
                        self.checkers[y, x, y_channel, x_channel] = True
                y_channel += 1
            x_channel += 1

    def mask_reduce(self, mask):
        mask = cv2.resize(mask, dsize=(self.size, self.size), interpolation=cv2.INTER_NEAREST)

        cells = np.max(mask)
        newmask = np.zeros((cells, self.dim, self.dim), dtype=bool)

        # For each cell ID in the mask
        for cell_id in range(1, cells + 1):
            cell_mask = ma.masked_not_equal(mask, cell_id)
            x1, x2, y1, y2 = extract_bboxes(~cell_mask.mask)

            min_x = x1 // self.fact
            max_x = min(self.dim, x2 // self.fact + 1)
            min_y = y1 // self.fact
            max_y = min(self.dim, y2 // self.fact + 1)

            for w in range(min_x, max_x):
                for h in range(min_y, max_y):
                    if np.any(np.logical_and(~cell_mask.mask, self.checkers[:, :, h, w])):
                        newmask[cell_id - 1, h, w] = True
        return newmask
