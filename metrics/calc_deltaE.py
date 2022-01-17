import numpy as np
from colour import models
import colour


def calc_deltaE(source, target, color_space, method='CIE 2000', chromatic_adaptation_transform='CAT02'):
    # type: (numpy.ndarray, numpy.ndarray, str, str) -> float
    # method = 'CIE 2000' | 'CIE 1976'
    
    assert isinstance(color_space, str), "color_space should be string"
    COLORSPACE_DICT = {'ProPhotoRGB': models.RGB_COLOURSPACE_PROPHOTO_RGB,
                       'sRGB': models.RGB_COLOURSPACE_sRGB,
                       'AdobeRGB': models.RGB_COLOURSPACE_ADOBE_RGB1998,
                       'DisplayP3': models.RGB_COLOURSPACE_DISPLAY_P3,
    }
    assert COLORSPACE_DICT.get(color_space) != None, "color_space should be ProPhotoRGB, sRGB, AdobeRGB, DisplayP3"
    color_space = COLORSPACE_DICT.get(color_space)
    source = np.reshape(source, (-1,3))
    target = np.reshape(target, (-1,3))
    source_XYZ = models.RGB_to_XYZ(
        source,
        color_space.whitepoint,
        color_space.whitepoint,
        color_space.matrix_RGB_to_XYZ,
        chromatic_adaptation_transform=chromatic_adaptation_transform
    )
    target_XYZ = models.RGB_to_XYZ(
        target,
        color_space.whitepoint,
        color_space.whitepoint,
        color_space.matrix_RGB_to_XYZ,
        chromatic_adaptation_transform=chromatic_adaptation_transform
    )
    source_Lab = models.XYZ_to_Lab(
        source_XYZ,
        color_space.whitepoint,
    )
    target_Lab = models.XYZ_to_Lab(
        target_XYZ,
        color_space.whitepoint,
    )    
    deltaE = colour.delta_E(source_Lab, target_Lab, method=method)
    # if source.shape[0] == 1:
    #     return deltaE
    # deltaE = sum(deltaE)/deltaE.shape[0]
    return np.mean(deltaE)
