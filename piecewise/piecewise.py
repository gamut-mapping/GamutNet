# -*- coding: utf-8 -*-
import numpy as np


def make_mask_piecewise(x, bins=None):
    # the input must have its channel axis
    assert (3 == len(x.shape))

    # we handle either 8bpc or 16bpc images
    assert x.dtype in (np.uint8, np.uint16)

    # set the default bins
    if bins is None:
        # either [0, 1); [1, 255); [255, infinity),
        # or     [0, 1); [1, 65535); [65535, infinity).
        bins = [0, 1, np.iinfo(x.dtype).max]

    height, width, channels = x.shape
    dims = tuple([len(bins)] * channels)  # the bin-dimensions
    multi_indices = np.digitize(x, bins, right=False) - 1  # the multi-dimensional indices (zero-based)
    indices = np.ravel_multi_index(multi_indices.reshape(-1, channels).T, dims)  # the raveled indices
    pw_mask = indices.reshape(height, width).astype(np.int)
    return pw_mask


def fit_mapping_piecewise(x_img, y_img, pw_mask, kernel):
    mappings = list()
    for mask_value in np.unique(pw_mask):
        mask = pw_mask == mask_value
        x, y = x_img[mask], y_img[mask]
        mapping = np.linalg.lstsq(kernel(x), y, rcond=None)[0]
        mappings.append(mapping)
    pw_mapping = np.transpose(np.dstack(mappings), (2, 0, 1))
    return pw_mapping


def apply_mapping_piecewise(x_img, pw_mapping, pw_mask, kernel,
                            skip=(0, 13, 26)):  # skip *in-gamut* cases of 27-pieces
    # copy the input as-is because some cases could be skipped.
    prediction = x_img.copy()
    for mask_value in np.unique(pw_mask):
        if (skip is not None) and (mask_value in skip):
            continue  # skip a given set of mask_values
        mapping = pw_mapping[mask_value]
        mask = pw_mask == mask_value
        x = x_img[mask]
        y_hat = kernel(x).dot(mapping)
        prediction[mask] = np.clip(y_hat, 0, 1)
    return prediction
