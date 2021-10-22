import numpy as np

def normalize(img):
  min_ = np.min(img)
  if min_ < 0:
    img += -min_
  else:
    img -= min_
  max_ = np.max(img)
  img /= max_
  
  for k in range(img.shape[0]):
    for d in range(img.shape[1]):
      assert 0 <= img[k, d] <= 1
  return img