import numpy as np 

class DataLoaderFromArrays:
    
    def __init__(self, features, labels, shuffle=True, one_hot=True, normalization=True):
        self.N_instances = features.shape[0]
        if normalization:
            self._features = np.float32(features/255.)
        else:
            self._features = features
        if one_hot:
            onehot_labels = np.zeros((self.N_instances, labels.max()+1))
            onehot_labels[np.arange(self.N_instances),labels] = 1
            self._labels = onehot_labels
        else:
            self._labels = labels
        self._shuffle = shuffle
        if self._shuffle:
            self.shuffling()
        self._pos = 0

    def next_batch(self, batch_size):
        if self._pos+batch_size >= self.N_instances:
            batch = (self._features[self._pos:self.N_instances], self._labels[self._pos:self.N_instances])
            self._pos = 0
            if self._shuffle:
                self.shuffling()
            return batch
        batch = (self._features[self._pos:self._pos+batch_size], self._labels[self._pos:self._pos+batch_size])
        self._pos += batch_size
        return batch
    
    def get_data(self):
        return self._features, self._labels

    def shuffling(self):
        indices = np.arange(0, self.N_instances)  # get all possible indexes
        np.random.shuffle(indices)  # shuffle indexes
        self._features = self._features[indices]
        self._labels = self._labels[indices]

    def normalize(self, data):
        return np.float32(data/255.)
    
    def one_hot_encoding(self, labels):
        onehot_labels = np.zeros((labels.shape[0], labels.max()+1))
        onehot_labels[np.arange(labels.shape[0]),labels] = 1
        return onehot_labels