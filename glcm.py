from skimage.feature import greycomatrix, greycoprops
from fishes import test_image
import mahotas as mh
import cv2

# the GLCM features computed by Spampinato et al. are Haralick features: Haralick et al. in "Textural Features for Image Classification"
# given an input image, the mahotas library can compute the GLCM and these features for us: https://github.com/luispedro/mahotas/blob/master/mahotas/features/texture.py
# skimage greycoprops() only computes 6: https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.greycoprops


def skimage_glcmFeatures(img):
    glcmFeatures = {}

    glcm = greycomatrix(img, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)

    glcm['correlation'] = greycoprops(glcm, 'correlation')[0][0]
    glcm['energy'] = greycoprops(glcm, 'energy')[0][0]

    return glcmFeatures


def mahotas_glcmFeatures(grayImg):
    # a 4x14 feature vector. 4, for 4 directions. 14, for 14 features listed in Haralick's paper.
    textures = mh.features.haralick(grayImg, compute_14th_feature=True)
    return ('glcmFeatures', textures.mean(axis=0))


if __name__ == "__main__":
    img = test_image(True)
    glcm = mahotas_glcmFeatures(img)
    print('hello')
