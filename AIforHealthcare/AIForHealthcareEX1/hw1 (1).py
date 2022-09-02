import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale, iradon_sart
import argparse
import cv2
from pathlib import Path

def create_sinograms(args):

    shepp_logan = shepp_logan_phantom()
    shepp_logan = rescale(shepp_logan, scale=0.4, mode='reflect', channel_axis=None)

    circle = cv2.imread(str(Path(args.path_circle)), cv2.IMREAD_GRAYSCALE)
    circle = rescale(circle, scale=0.4, mode='reflect', channel_axis=None)

    rectangle = cv2.imread(str(Path(args.path_rectangle)), cv2.IMREAD_GRAYSCALE)
    rectangle = rescale(rectangle, scale=0.4, mode='reflect', channel_axis=None)

    fig, (ax11, ax22, ax33) = plt.subplots(1, 3, figsize=(8, 4.5))

    ax11.set_title("shepp_logan")
    ax11.imshow(shepp_logan, cmap=plt.cm.Greys_r)

    ax22.set_title("circle")
    ax22.imshow(circle, cmap=plt.cm.Greys_r)

    ax33.set_title("rectangle")
    ax33.imshow(rectangle, cmap=plt.cm.Greys_r)
    plt.show()

    results = []
    for image, name in [(shepp_logan, 'sl'), (circle, 'circle'), (rectangle, 'rectangle')]:

        if args.number_of_angels > max(image.shape):
            args.number_of_angels = max(image.shape)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

        ax1.set_title("Original")
        ax1.imshow(image, cmap=plt.cm.Greys_r)

        theta = np.linspace(start=0., stop=180., num=args.number_of_angels, endpoint=False)

        if name == 'circle':
            sinogram = radon(image, theta=theta, circle=True)
        else:
            sinogram = radon(image, theta=theta)

        results.append([sinogram, theta, image, name])
        dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
        ax2.set_title(f"Radon transform\n(Sinogram)\n"
                      f"number of angels:{args.number_of_angels}")
        ax2.set_xlabel("Projection angle (deg)")
        ax2.set_ylabel("Projection position (pixels)")
        ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
                   extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
                   aspect='auto')

        fig.tight_layout()
        plt.show()

    return results
def fbp_reconstruction(args, sinograms):
    """
    :param args:
    :type args:
    :param sinograms:
    :type sinograms:
    :return:
    :rtype:
    """

    for sinogram, theta, image, name in sinograms:

        if name == 'circle':
            reconstruction_fbp = iradon(sinogram, theta=theta, filter_name=args.filter, circle=True)
            error = reconstruction_fbp - image
        else:
            reconstruction_fbp = iradon(sinogram, theta=theta, filter_name=args.filter)
            error = reconstruction_fbp - image

        print(f'FBP rms reconstruction error: {np.sqrt(np.mean(error ** 2)):.3g}')

        imkwargs = dict(vmin=-0.2, vmax=0.2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                                       sharex=True, sharey=True)
        ax1.set_title("Reconstruction\nFiltered back projection")
        ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
        ax2.set_title("Reconstruction error\nFiltered back projection")
        ax2.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
        plt.show()
def sart_reconstruction(args, sinograms):

    for sinogram, theta, image, _ in sinograms:

        for iteration in range(args.num_iterations_in_sart_reco):

            if iteration == 0:
                reconstruction_sart = iradon_sart(sinogram, theta=theta)

            else:
                reconstruction_sart = iradon_sart(
                    sinogram,
                    theta=theta,
                    image=reconstruction_sart
                )

            error = reconstruction_sart - image
            print(f'SART iteration: {iteration} rms reconstruction error: '
                  f'{np.sqrt(np.mean(error ** 2)):.3g}')

        fig, axes = plt.subplots(1, 2, figsize=(8, 8.5), sharex=False, sharey=False)
        ax = axes.ravel()

        ax[0].set_title(f"Reconstruction\nSART\n iterations:{iteration+1}")
        ax[0].imshow(reconstruction_sart, cmap=plt.cm.Greys_r)

        plt.show()
def main(args):
    sinograms = create_sinograms(args)
    fbp_reconstruction(args, sinograms)
    sart_reconstruction(args, sinograms)

if __name__ == '__main__':
    __file__ = 'main.py'

    parser = argparse.ArgumentParser(description='ex1')
    "================================================================================="
    parser.add_argument('--number_of_angels', type=int, default=90,
                        help='projection angels, can be [18, 24, 90, ...]')
    parser.add_argument('--path_rectangle', type=str, default=f'images/rectangle.jpg')
    parser.add_argument('--path_circle', type=str, default=f'images/circle.jpg')
    parser.add_argument('--filter', type=str, default=None,
                        help="can be: ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'], None for no filter")
    parser.add_argument('--num_iterations_in_sart_reco', type=int, default=4)
    "================================================================================="
    args = parser.parse_known_args()[0]
    "================================================================================="
    main(args)