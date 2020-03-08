import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from random_dots import random_dots

# import an image as grayscale 2D array
left = cv2.imread('RandomA.png', 0).tolist()
right = cv2.imread('RandomB.png', 0).tolist()


def matching_cost(L, R):
    mean = int((L + R) // 2)
    return ((mean - L) * (mean - R)) // 2


def cost_matrix(L, R):
    occlusion = 3.8
    cost_mat = [[0] * (len(L) + 1) for _ in range(len(R) + 1)]
    path = []

    for i in range(len(cost_mat)):
        cost_mat[i][0] = i * occlusion
        cost_mat[0][i] = i * occlusion

    for i in range(len(cost_mat)):
        for j in range(len(cost_mat)):
            case_1 = cost_mat[i - 1][j - 1] + matching_cost(L[i - 1], R[j - 1])
            case_2 = cost_mat[i - 1][j] + occlusion
            case_3 = cost_mat[i][j - 1] + occlusion
            cost_mat[i][j] = min(case_1, case_2, case_3)

            if cost_mat[i][j] == case_1:
                path.append([-1, -1])
            elif cost_mat[i][j] == case_2:
                path.append([-1, 0])
            else:
                path.append([0, -1])

    # back tracking
    back_i = len(cost_mat) - 1
    back_j = len(cost_mat[0]) - 1

    result = []

    while back_i > 0 and back_j > 0:
        case_1 = cost_mat[back_i - 1][back_j - 1] + matching_cost(L[back_i - 1], R[back_j - 1])
        case_2 = cost_mat[back_i][back_j - 1] + occlusion
        case_3 = cost_mat[back_i - 1][back_j] + occlusion

        if cost_mat[back_i][back_j] == case_1:
            result.append([back_i - 1, back_j - 1, 255])
            back_i -= 1
            back_j -= 1
        elif cost_mat[back_i][back_j] == case_2:
            result.append([back_i, back_j - 1, 0])
            back_j -= 1
        else:
            result.append([back_i - 1, back_j, 0])
            back_i -= 1

    result = result[::-1]
    return result    # [i, j, RGB]

# create two random dot images
random_dots()

res_matrix = []
for row in range(len(left)):
    L = left[row]
    R = right[row]
    res_matrix.append(cost_matrix(L, R))
plt.imshow(res_matrix)
plt.show()

