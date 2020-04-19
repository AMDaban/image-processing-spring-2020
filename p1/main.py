import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_image(n):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    image_path = f"{dir_path}/images/{n}.jpg"

    return cv2.imread(image_path, 0)


def dft_trans(img):
    trans_img = np.fft.fft2(img)
    shifted_trans_img = np.fft.fftshift(trans_img)
    magnitude_spectrum = np.log(1 + np.abs(shifted_trans_img))
    return shifted_trans_img, magnitude_spectrum


def idft_trans(img):
    shifted_trans = np.fft.ifftshift(img)
    return np.fft.ifft2(shifted_trans).real.clip(0, 255).astype(np.uint8)


def plot(img):
    plt.imshow(img)
    plt.show()


def image_1():
    img = get_image(1)
    plot(img)

    shifted_trans_img, magnitude_spectrum = dft_trans(img)
    plot(magnitude_spectrum)

    width, height = img.shape
    mask = np.ones(img.shape, dtype=np.uint8)
    for i in range(width):
        if np.mean(magnitude_spectrum[i, :]) >= 9.:
            mask[i, :] = 0
    for j in range(height):
        if np.mean(magnitude_spectrum[:, j]) >= 9.:
            mask[:, j] = 0
    mask[width // 2 - 5:width // 2 + 5, height // 2 - 5:height // 2 + 5] = 1

    plot(mask * magnitude_spectrum)

    filtered_trans = mask * shifted_trans_img
    inv_img = cv2.GaussianBlur(idft_trans(filtered_trans), (5, 5), 0)

    plot(inv_img)


def image_2():
    pass


def image_3():
    pass


def image_4():
    pass


def image_5():
    pass


def image_6():
    pass


def image_7():
    pass


def image_8():
    pass


def image_9():
    pass


def image_10():
    pass


def image_11():
    pass


def image_12():
    pass


def main():
    image_1()
    image_2()
    image_3()
    image_4()
    image_5()
    image_6()
    image_7()
    image_8()
    image_9()
    image_10()
    image_11()
    image_12()


if __name__ == '__main__':
    main()
