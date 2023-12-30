import numpy as np
from PIL import Image
import sys
path = sys.argv[1]
type = sys.argv[2]
# Load the NPZ file
if type == 'samples':
    data = np.load(path+'/samples_500x32x32x3.npz')['arr_0']
else:
    data = np.load(path+'/corrupted_samples_500x32x32x3.npz')['arr_0']

# Iterate over the data and save each image
for i, img_data in enumerate(data):
    print(i)
    img = Image.fromarray(img_data.astype(np.uint8))
    if type == 'samples':
        img.save(f'{path}/image_{i}.png')
    else:
        img.save(f'{path}/corrupted_image_{i}.png')
        

