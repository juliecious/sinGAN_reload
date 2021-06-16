import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import pyramid_gaussian

logo = io.imread('http://scikit-image.org/_static/img/logo.png')

plt.figure (figsize = (4, 4))
plt.imshow(logo)
plt.axis('off')
plt.tight_layout()
plt.show()
io.imsave('local_logo.png', logo)


rows, cols, dim = logo.shape
pyramid = tuple (pyramid_gaussian(logo, downscale =2, channel_axis=- 1))

composite_logo = np.zeros((rows, cols + cols // 2, 3), dtype = np.double)
composite_logo[:rows, :cols, :] = pyramid[0]

i_row = 0

for p in pyramid[1:]:
    n_rows, n_cols =p.shape[:2]
    composite_logo[i_row: i_row + n_rows, cols:cols + n_cols]= p
    i_row += n_rows

fig, ax = plt.subplots()
ax.imshow(composite_logo)
plt.show() 