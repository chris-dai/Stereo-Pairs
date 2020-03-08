from PIL import Image
import numpy as np
import random

def randomNum():
    num = random.randint(0,1)
    if(num == 1):
        i = 255
    else:
        i = 0
    return i


def random_dots():
    imageA = np.array([[randomNum() for _ in range(512)] for _ in range(512)])
    imageB = np.array([[randomNum() for _ in range(256)] for _ in range(256)])

    l = np.copy(imageA)
    l[128:128 + 256, 124:124 + 256] = np.copy(imageB)
    r = np.copy(imageA)
    r[128:128 + 256, 132:132 + 256] = np.copy(imageB)

    left = l.tolist()
    right = r.tolist()

    imgLeft = Image.new("L", (512, 512))
    imgLeft.putdata(np.reshape(l, 512 * 512))
    imgRight = Image.new("L", (512, 512))
    imgRight.putdata(np.reshape(r, 512 * 512))
    imgLeft.save("RandomA.png")
    imgRight.save("RandomB.png")

random_dots()