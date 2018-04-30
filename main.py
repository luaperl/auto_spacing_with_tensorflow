from utils import *
from model import Model2

if __name__ == '__main__':
    train_data = DataLoader('../data/trainX.txt', '../data/trainY.txt')
    test_data = DataLoader('../data/testX.txt', '../data/testY.txt')

    train_data.set_batch(100)
    test_data.set_batch(100)

    char_dic = CharDic([train_data])

    model = Model2(train_data=train_data,
                  test_data=test_data,
                  char_dic=char_dic,
                  model_name='bilstm_crf_n3_e300_h2002')

    model.train()
    model.test()