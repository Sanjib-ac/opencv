"""Resize images"""

import cv2

file = r'C:\Program Files\National Instruments\LabVIEW 2022\examples\Ngene' \
       r'\Deep Learning Toolkit\Object_Detection\samples\nut.png'

img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

down_width = 32
down_height = 32
down_points = (down_width, down_height)
resized_down = cv2.resize(img, down_points, interpolation=cv2.INTER_AREA)

cv2.imshow('image', img)
cv2.imshow('mod', resized_down)
print(f'original size: {img.shape}, resized: {resized_down.shape}')

# save image
status = cv2.imwrite(r'C:\Program Files\National Instruments\LabVIEW 2022\examples\Ngene'
                     r'\Deep Learning Toolkit\Object_Detection\samples\n_32.png', resized_down)

print(f'Image written to file-system : {status}')
cv2.waitKey(0)
cv2.destroyAllWindows()
