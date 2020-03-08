import cv2
img = cv2.imread('view2.png', 0).tolist()
print(img)

import matplotlib.pyplot as plt

plt.imshow(img, cmap='gray')
plt.show()