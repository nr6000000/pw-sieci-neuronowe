import pickle
import random
import struct
from array import array
from pathlib import Path

import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from activations import Activation
from losses import Loss
from matplotlib import pyplot as plt
from network import Layer, MLPNetwork


def load_mnist(training_images_path,training_labels_path,
                test_images_path, test_labels_path):
    def read_images_labels(images_path, labels_path):        
        labels = []
        with open(labels_path, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = np.array(array("B", file.read()))     
        
        with open(images_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = np.empty((size, rows*cols))
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            images[i,:] = img
        
        return images, labels
            
    X_train, y_train = read_images_labels(training_images_path, training_labels_path)
    X_test, y_test = read_images_labels(test_images_path, test_labels_path)
    return X_train, y_train, X_test, y_test

base_path = Path('dataset/mnist/')
training_images_path = base_path / 'train-images-idx3-ubyte/train-images-idx3-ubyte'
training_labels_path = base_path / 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'
test_images_path = base_path / 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
test_labels_path = base_path / 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

X_train, y_train, X_test, y_test = load_mnist(training_images_path, training_labels_path,
                                              test_images_path, test_labels_path)

def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols)
    plt.figure(figsize=(8,4))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0].reshape(28, 28)
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 8)
        index += 1

def onehot_encode(arr):
    encoded = np.zeros((len(arr), 10))
    for i, label in enumerate(arr):
        encoded[i, label] = 1
    return encoded

def onehot_decode(arr):
    return np.argmax(arr, axis=1)

y_train = onehot_encode(y_train)
y_test = onehot_encode(y_test)

X_train /= 255
X_test /= 255

model = MLPNetwork(
    784,
    layers=[
        Layer(512, Activation.relu, Activation.Gradient.relu),
        Layer(256, Activation.relu, Activation.Gradient.relu),
        Layer(10, Activation.softmax, Activation.Gradient.softmax),
    ],
    loss_function=Loss.cce,
    loss_gradient_function=Loss.Gradient.cce,
    include_bias=True,
    seed=42,
)

def acc(y_test, y_pred):
    correct = np.equal(
        onehot_decode(y_test), 
        onehot_decode(y_pred)
    )
    return np.count_nonzero(correct)/len(correct)

# model.fit(X_train, y_train, 256, 
#           save_metric_history=True, save_loss_history=True, metric=acc,
#           X_test=X_test, y_test=y_test)

# with open('mnist_history.pkl', 'wb') as file:
#     pickle.dump(model, file)

with open('mnist_history.pkl', 'rb') as file:
    model = pickle.load(file)

y_pred = model.predict(X_test)

print(acc(y_test, y_pred))

y_train_decoded = onehot_decode(y_train)
y_test_decoded = onehot_decode(y_test)

images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(X_train[r])
    titles_2_show.append(f'train[{r}] = {y_train_decoded[r]}')    

# for i in range(0, 5):
#     r = random.randint(1, 10000)
#     images_2_show.append(X_test[r])        
#     titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

show_images(images_2_show, titles_2_show)
plt.tight_layout()
plt.show()

sns.lineplot({
    'Train': model.loss_history,
    'Test': model.test_loss_history,
})
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.yscale('log')
plt.show()

sns.lineplot({
    'Train': np.array(model.metric_history)*100,
    'Test': np.array(model.test_metric_history)*100,
})
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()

y_pred_decoded = onehot_decode(y_pred)
wrong_indices = np.argwhere(np.not_equal(y_pred_decoded, y_test_decoded))
wrong_indices = np.squeeze(wrong_indices)

images_2_show = []
titles_2_show = []
for i in np.random.choice(wrong_indices, 10):
    images_2_show.append(X_test[i])
    titles_2_show.append(f'test[{i}] = {y_pred_decoded[i]}')    
show_images(images_2_show, titles_2_show)
plt.tight_layout()
plt.show()
