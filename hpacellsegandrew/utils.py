"""Utility functions for the HPA Cell Segmentation package."""
import numpy as np
import scipy.ndimage as ndi
from skimage import filters, measure, segmentation
from skimage.morphology import (closing, disk, binary_erosion,
                                remove_small_holes, remove_small_objects)
import cv2

HIGH_THRESHOLD = 0.4
LOW_THRESHOLD = HIGH_THRESHOLD - 0.25

def __fill_holes(image):
    """Fill_holes for labelled image, with a unique number."""
    boundaries = segmentation.find_boundaries(image)
    image = np.multiply(image, np.invert(boundaries))
    image = ndi.binary_fill_holes(image > 0)
    image = ndi.label(image)[0]
    return image

def label_nuclei(nuclei_pred, scale_factor=1.0):
    """Return the labeled nuclei mask data array.

    This function works best for Human Protein Atlas cell images with
    predictions from the CellSegmentator class.

    Keyword arguments:
    nuclei_pred -- a 3D numpy array of a prediction from a nuclei image.

    Returns:
    nuclei-label -- An array with unique numbers for each found nuclei
                    in the nuclei_pred. A value of 0 in the array is
                    considered background, and the values 1-n is the
                    areas of the cells 1-n.
    """
    scale_sq = scale_factor**2
    img_copy = np.copy(nuclei_pred[..., 2])
    borders = (nuclei_pred[..., 1] > 0.05).astype(np.uint8)
    m = img_copy * (1 - borders)

    img_copy[m <= LOW_THRESHOLD] = 0
    img_copy[m > LOW_THRESHOLD] = 1
    img_copy = img_copy.astype(np.bool)
    img_copy = binary_erosion(img_copy)
    # TODO: Add parameter for remove small object size for
    #       differently scaled images.
    # img_copy = remove_small_objects(img_copy, 500)
    img_copy = img_copy.astype(np.uint8)
    markers = measure.label(img_copy).astype(np.uint32)

    mask_img = np.copy(nuclei_pred[..., 2])
    mask_img[mask_img <= HIGH_THRESHOLD] = 0
    mask_img[mask_img > HIGH_THRESHOLD] = 1
    mask_img = mask_img.astype(np.bool)
    mask_img = remove_small_holes(mask_img, int(1000 * scale_sq))
    # TODO: Figure out good value for remove small objects.
    # mask_img = remove_small_objects(mask_img, 8)
    mask_img = mask_img.astype(np.uint8)
    nuclei_label = segmentation.watershed(
        mask_img, markers, mask=mask_img, watershed_line=True
    )
    nuclei_label = remove_small_objects(nuclei_label, int(2500 * scale_sq))
    nuclei_label = measure.label(nuclei_label)

    nuclei_label = resize_label(nuclei_label, scale_factor)

    return nuclei_label


def label_cell(nuclei_pred, cell_pred, return_nuclei_label=True, scale_factor=1.0):
    """Label the cells and the nuclei.

    Keyword arguments:
    nuclei_pred -- a 3D numpy array of a prediction from a nuclei image.
    cell_pred -- a 3D numpy array of a prediction from a cell image.

    Returns:
    A tuple containing:
    nuclei-label -- A nuclei mask data array.
    cell-label  -- A cell mask data array.

    0's in the data arrays indicate background while a continous
    strech of a specific number indicates the area for a specific
    cell.
    The same value in cell mask and nuclei mask refers to the identical cell.

    NOTE: The nuclei labeling from this function will be sligthly
    different from the values in :func:`label_nuclei` as this version
    will use information from the cell-predictions to make better
    estimates.
    """
    scale_sq = scale_factor**2
    def __wsh(
        mask_img,
        threshold,
        border_img,
        seeds,
        threshold_adjustment=0.35,
        small_object_size_cutoff=int(10*scale_sq)
        #small_object_size_cutoff=int(10)
    ):
        img_copy = np.zeros_like(mask_img)
        m = seeds * border_img  # * dt
        img_copy[m > threshold + threshold_adjustment] = 1
        img_copy = img_copy.astype(np.bool)
        img_copy = remove_small_objects(img_copy, small_object_size_cutoff).astype(
            np.uint8
        )

        mask_img = np.where(mask_img <= threshold, 0, 1)
        mask_img = mask_img.astype(np.bool)
        #mask_img = remove_small_holes(mask_img, int(63)) # Significantly different from original
        mask_img = remove_small_holes(mask_img, int(1000*scale_sq))
        mask_img = remove_small_objects(mask_img, int(8*scale_sq)).astype(np.uint8)
        markers = ndi.label(img_copy, output=np.uint32)[0]
        labeled_array = segmentation.watershed(
            mask_img, markers, mask=mask_img, watershed_line=True
        )
        return labeled_array

    nuclei_label = __wsh(
        nuclei_pred[..., 2] / 255.0,
        0.4,
        1 - (nuclei_pred[..., 1] + cell_pred[..., 1]) / 255.0 > 0.05,
        nuclei_pred[..., 2] / 255,
        threshold_adjustment=-0.25,
        #small_object_size_cutoff=int(32), # Significantly different
        small_object_size_cutoff=int(500*scale_sq)
    )

    # for hpa_image, to remove the small pseduo nuclei
    #nuclei_label = remove_small_objects(nuclei_label, int(157))
    nuclei_label = remove_small_objects(nuclei_label, int(2500*scale_sq))
    nuclei_label = measure.label(nuclei_label)
    # this is to remove the cell borders' signal from cell mask.
    # could use np.logical_and with some revision, to replace this func.
    # Tuned for segmentation hpa images
    threshold_value = max(0.22, filters.threshold_otsu(cell_pred[..., 2] / 255) * 0.5)
    # exclude the green area first
    cell_region = np.multiply(
        cell_pred[..., 2] / 255 > threshold_value,
        np.invert(np.asarray(cell_pred[..., 1] / 255 > 0.05, dtype=np.int8)),
    )
    sk = np.asarray(cell_region, dtype=np.int8)
    distance = np.clip(cell_pred[..., 2], 255 * threshold_value, cell_pred[..., 2])
    cell_label = segmentation.watershed(-distance, nuclei_label, mask=sk)
    cell_label_copy = cell_label.copy()
    #cell_label = remove_small_objects(cell_label, int(344)).astype(np.uint8)
    cell_label = remove_small_objects(cell_label, int(5500*scale_sq)).astype(np.uint8)
    selem = disk(6)
    cell_label = closing(cell_label, selem)
    cell_label = __fill_holes(cell_label)
    # this part is to use green channel, and extend cell label to green channel
    # benefit is to exclude cells clear on border but without nucleus
    sk = np.asarray(
        np.add(
            np.asarray(cell_label > 0, dtype=np.int8),
            np.asarray(cell_pred[..., 1] / 255 > 0.05, dtype=np.int8),
        )
        > 0,
        dtype=np.int8,
    )
    cell_label = segmentation.watershed(-distance, cell_label, mask=sk)
    cell_label = __fill_holes(cell_label)
    cell_label = np.asarray(cell_label > 0, dtype=np.uint8)
    cell_label = measure.label(cell_label)
    #cell_label = remove_small_objects(cell_label, int(344))
    cell_label = remove_small_objects(cell_label, int(5500*scale_sq))
    cell_label = measure.label(cell_label)
    cell_label = np.asarray(cell_label, dtype=np.uint16)

    if not return_nuclei_label:
        return cell_label
    nuclei_label = np.multiply(cell_label > 0, nuclei_label) > 0
    nuclei_label = measure.label(nuclei_label)
    #nuclei_label = remove_small_objects(nuclei_label, int(2500*scale_sq))
    # This removal is a bug which causes some cells to have no nuclei !!!

    #nuclei_label = remove_small_objects(nuclei_label, int(157))
    nuclei_label = np.multiply(cell_label, nuclei_label > 0)

    nuclei_label = resize_label(nuclei_label, scale_factor)
    cell_label = resize_label(cell_label, scale_factor)

    return nuclei_label, cell_label

def resize_label(label, scale_factor=1.0):
    if scale_factor != 1.0:
        # Find out correct size
        small = label.shape[0]
        if small / scale_factor > 4000:
            size = 4096
        elif small / scale_factor > 3000:
            size = 3072
        elif small / scale_factor > 2000:
            size = 2048
        else:
            size = 1728
        #label[..., 0] = 0
        label = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
    return label

