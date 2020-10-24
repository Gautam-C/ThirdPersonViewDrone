import cv2
import numpy as np

img = cv2.imread('opencv-logo-white.png')

green = img[:, :, 1]

cv2.imshow('green', green)

lim = [[1, 2, 3], [4, 5, 6]]

new = [1, 2]

one, two = new
print(one)
print(two)

x, y = lim

print(x)
print(y)

for ind, (x, y, z) in enumerate(lim):
    print(f'the value at {ind} is {x}, {y}, {z}')

print(len(lim[0]))

while True:
    if 27 == cv2.waitKey() & 0xFF:
        break
cv2.destroyAllWindows()