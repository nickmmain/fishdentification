import json
from fishes import test_image
from glcm import mahotas_glcmFeatures
from gabor import gaborFeatures
from histogram import spampinatoHistogramFeatures

img = test_image(True)
features = {}
features['texture'] = {}

gf = gaborFeatures(img)
features['texture'].update(gf)

glcm = mahotas_glcmFeatures(img)
features['texture'][glcm[0]] = glcm[1]

histo = spampinatoHistogramFeatures(img)
features['texture'][histo[0]] = histo[1]
