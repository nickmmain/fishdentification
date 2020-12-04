# https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.greycomatrix
# https://www.mathworks.com/help/images/texture-analysis-using-the-gray-level-co-occurrence-matrix-glcm.html
from skimage.feature import greycomatrix, greycoprops
from fishes import test_image

# using skimage greycoprops we can take care of
# energy, correlation, ASM, homogeneity, dissimilarity, and contrast
# https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.greycoprops

# Leaving us with these to take care of:
# Inertia, Entropy, Inverse Difference Moment, Sum Average,
# Sum Variance, Sum Entropy, Difference Average, Difference
# Variance, Difference Entropy, Information measure of correlation 1, Information measure of correlation 2, Maximal Correlation Coefficient.


def glcmFeatures(img):
    glcmFeatures = {}

    glcm = greycomatrix(img, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)

    glcm['correlation'] = greycoprops(glcm, 'correlation')[0][0]
    glcm['energy'] = greycoprops(glcm, 'energy')[0][0]

    return glcmFeatures


if __name__ == "__main__":
    img = test_image(True)
    hello = glcm(img)
