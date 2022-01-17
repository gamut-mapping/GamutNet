# -*- coding: utf-8 -*-
import numpy as np


def generate_patch(mask, size, random_generator=None):
    # patch dimensions
    rows, cols = size
    half_rows = int(rows / 2)
    half_cols = int(cols / 2)

    # crop the mask according to the given patch dimensions
    cropped_mask = mask[half_rows:-half_rows, half_cols:-half_cols]

    # find pixels where the mask value is True
    nonzero_indices = np.flatnonzero(cropped_mask)
    if random_generator is not None:
        assert isinstance(random_generator, np.random.Generator)
        # if a random number generator is provided, it shuffles the indices
        random_generator.shuffle(nonzero_indices)
    nonzero_multi_indices = np.unravel_index(nonzero_indices, cropped_mask.shape)

    # generate patches
    for i, j in zip(*nonzero_multi_indices):
        # The first 2-tuple is the center point (row_center, col_center); and
        # The second 4-tuple is the bounding box (top, bottom, left, right).
        # The center point is for indexing the groundtruth color.
        #   e.g. y[row_center, col_center, :]
        # The bounding box is for indexing the input patch.
        #   e.g. x[top:bottom, left:right, :]
        # The below line yields ((row_center, col_center), (top, bottom, left, right)).
        yield (i + half_rows, j + half_cols), (i, i + rows, j, j + cols)


def get_pad_width(patch_size):
    # pad_width := ((before_axis_0, after_axis_0), (before_axis_1, after_axis_1), (before_axis_2, after_axis_2))
    half_rows, half_cols = tuple(int(s / 2) for s in patch_size)
    pad_width = ((half_rows, half_rows), (half_cols, half_cols), (0, 0))
    return pad_width
