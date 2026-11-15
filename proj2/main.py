import numpy as np
from hopfield import HopfieldNet
from matplotlib import animation
from matplotlib import pyplot as plt


def main():
    hopfield = HopfieldNet(
        False,
        'hebb',
        100,
        0.01,
        10
    )

    width = 14
    height = 9

    X = np.genfromtxt('dane/animals-14x9.csv', delimiter=',')
    hopfield.fit(X)

    plt.imshow(X[3].reshape(height, width))
    plt.show()

    noise_indices = np.random.choice(np.arange(X.shape[1]), int(0.1*X.shape[1]), replace=False)
    x_noisy = X[3]
    # x_noisy[noise_indices] = np.random.normal(0, 1/3, noise_indices.size)
    x_noisy[noise_indices] = -x_noisy[noise_indices]

    plt.imshow(x_noisy.reshape(height, width))
    plt.show()

    steps = hopfield.predict(x_noisy, save_steps=True)
    pred = steps[-1]

    fig, ax = plt.subplots()
    ims = []
    ax.imshow(steps[0].reshape(height, width))
    for step in steps:
        im = ax.imshow(step.reshape(height, width), animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=100)
    plt.show()

    plt.imshow(pred.reshape(height, width))
    plt.show()

if __name__ == '__main__':
    main()