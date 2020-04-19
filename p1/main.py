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
    plt.imshow(img, cmap="gray")
    plt.show()


def gaussian_lowpass_filter(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


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
    img = get_image(2)
    plot(img)

    shifted_trans_img, magnitude_spectrum = dft_trans(img)
    plot(magnitude_spectrum)

    width, height = img.shape
    mask = np.ones(img.shape, dtype=np.uint8)
    mask[0:width // 2 - 5, height // 2 - 2:height // 2 + 2] = 0
    mask[width // 2 + 5:width, height // 2 - 2:height // 2 + 2] = 0

    plot(magnitude_spectrum * mask)

    filtered_trans = mask * shifted_trans_img
    inv_img = cv2.blur(idft_trans(filtered_trans), (2, 2))

    plot(inv_img)


def image_3():
    img = get_image(3)
    plot(img)

    shifted_trans_img, magnitude_spectrum = dft_trans(img)
    plot(magnitude_spectrum)

    mask = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cv2.circle(mask, (96, 117), 10, (0, 0, 0), -1)
    cv2.circle(mask, (42, 85), 10, (0, 0, 0), -1)
    cv2.circle(mask, (204, 178), 10, (0, 0, 0), -1)
    cv2.circle(mask, (258, 210), 10, (0, 0, 0), -1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    plot(magnitude_spectrum * mask)

    filtered_trans = mask * shifted_trans_img
    inv_img = idft_trans(filtered_trans)

    plot(inv_img)


def image_4():
    pass


def image_5():
    img = get_image(5)
    plot(img)

    filtered_img = np.array(255 * (img / 255) ** 1.5, dtype=np.uint8)
    filtered_img = cv2.GaussianBlur(filtered_img, (5, 5), 0)

    plot(filtered_img)


def image_6():
    pass


def image_7():
    pass


def image_8():
    img = get_image(8)
    plot(img)

    filtered_img = np.array(255 * (img / 255) ** .5, dtype=np.uint8)
    filtered_img = cv2.medianBlur(filtered_img, 3)

    plot(filtered_img)


def image_9():
    pass


def image_10():
    pass


def image_11():
    img = get_image(11)
    plot(img)

    shifted_trans_img, magnitude_spectrum = dft_trans(img)
    plot(magnitude_spectrum)

    h = cv2.normalize(gaussian_lowpass_filter(img.shape, sigma=40), None, 0, 1, cv2.NORM_MINMAX)

    transformed_img = shifted_trans_img * h

    inv_image = idft_trans(transformed_img)

    plot(inv_image)


def image_12():
    img = get_image(12)
    plot(img)

    shifted_trans_img, magnitude_spectrum = dft_trans(img)
    plot(magnitude_spectrum)

    h = cv2.normalize(gaussian_lowpass_filter(img.shape, sigma=40), None, 0, 1, cv2.NORM_MINMAX)

    transformed_img = shifted_trans_img * h

    inv_image = idft_trans(transformed_img)

    plot(inv_image)


def main():
    image_1()
    image_2()
    image_3()
    # image_4()
    image_5()
    # image_6()
    # image_7()
    image_8()
    # image_9()
    # image_10()
    image_11()
    image_12()


if __name__ == '__main__':
    main()
