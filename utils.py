
import os
import pickle


class CharDic:
    def __init__(self, data_list, file_dir='./dic/word_dict.pickle'):

        self.UNKNOWN_TAG = 0
        self.word_to_ix = {}

        if os.path.isfile(file_dir):
            print('load CharDic...')

            with open(file_dir, 'rb') as handle:
                self.word_to_ix = pickle.load(handle)

        else:
            print('make CharDic...')

            for data_loader in data_list:
                for (x_data, y_data) in data_loader:
                    for sentence in x_data:
                        for word in sentence:
                            if word not in self.word_to_ix:
                                self.word_to_ix[word] = len(self.word_to_ix) + 1

            with open(file_dir, 'wb') as handle:
                pickle.dump(self.word_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.len = len(self.word_to_ix) + 1

    def __len__(self):
        return self.len

    def __getitem__(self, ch):
        if ch in self.word_to_ix:
            return self.word_to_ix[ch]
        else:
            return self.UNKNOWN_TAG



class DataLoader:
    def __init__(self, x_data_dir, y_data_dir, batch_size=5):
        self.x_data_dir = x_data_dir
        self.y_data_dir = y_data_dir

        self.batch_size = batch_size
        self.x_data = []
        self.y_data = []
        self.idx = 0

    def set_batch(self, batch_size):
        self.batch_size = batch_size

    def get_batch(self):
        return self.batch_size

    def _clear(self):
        self.x_data.clear()
        self.y_data.clear()
        self.idx = 0

    def __iter__(self):
        self._clear()

        with open(self.x_data_dir) as features, \
                open(self.y_data_dir) as labels:

            for sentence, label in zip(features, labels):
                if self.idx == self.batch_size:
                    yield self.x_data, self.y_data
                    self._clear()

                self.x_data.append(list(sentence[:-1]))
                self.y_data.append(list(label[:-1]))
                self.idx += 1

            else:
                yield self.x_data, self.y_data
