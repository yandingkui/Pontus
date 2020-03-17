from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Activation
from keras.datasets import imdb
from stringexperiment.pontus import pontus
from lstm_dga.match import lstm_getSingleFea,lstm_getAllFea
import random
from sklearn.metrics import accuracy_score,precision_score,recall_score


# mnist attention
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam




def getData(bfiles,AGDfile):
    p = pontus()
    root_dir = "../data_sets/"
    bfiles = "{}{}".format(root_dir, 'yd_20180427')
    AGDfile = "{}AGD{}.json".format(root_dir, 0)
    #aleax
    # bfiles="{}{}".format(root_dir,"Aleax")
    # AGDfile = "{}AGD{}.json".format(root_dir, 0)
    #wordlist
    # bfiles = "{}{}".format(root_dir, "Aleax2LD")
    # AGDfile = "{}wordlist.json".format(root_dir)
    trainDGADomain, testDGADomain, trainBenignDomain, testBenignDomain = p.getTrainTestDomains(benignFile=bfiles,
                                                                                               AGDfile=AGDfile)
    # trainDGADomain=trainDGADomain[:1]
    # testDGADomain=testDGADomain[:1]
    # trainBenignDomain=trainBenignDomain[:1]
    # testBenignDomain=testBenignDomain[:1]

    trainData = trainDGADomain + trainBenignDomain
    trainFeas= lstm_getAllFea(trainData)
    trainLabel = np.concatenate((np.ones(len(trainDGADomain)), np.zeros(len(trainBenignDomain))))
    testData = testDGADomain + testBenignDomain
    testFeas= lstm_getAllFea(testData)
    testLabel = np.concatenate((np.ones(len(testDGADomain)), np.zeros(len(testBenignDomain))))

    index = list(range(len(trainData)))
    random.shuffle(index)

    real_train_features = []
    real_train_labels = []
    for i in index:
        real_train_features.append(trainFeas[i])
        real_train_labels.append(trainLabel[i])

    index = list(range(len(testData)))
    random.shuffle(index)

    real_test_features = []
    real_test_labels = []
    for i in index:
        real_test_features.append(testFeas[i])
        real_test_labels.append(testLabel[i])

    return (np.array(real_train_features),np.array(real_train_labels)),(np.array(real_test_features),np.array(real_test_labels))

def saveVectors():
    root_dir = "../data_sets/"
    result_path="../result_data/bi/"
    bafiles=[("{}{}".format(root_dir, 'yd_20180427'),"{}AGD{}.json".format(root_dir, 0)),("{}{}".format(root_dir,"Aleax"),"{}AGD{}.json".format(root_dir, 0)),("{}{}".format(root_dir, "Aleax2LD"),"{}wordlist.json".format(root_dir))]

    for baf in bafiles:
        bfiles=baf[0]
        AGDfile=baf[1]
        saveFileName=bfiles[bfiles.rindex("/")+1:]
        (x_train, y_train), (x_test, y_test) = getData(bfiles,AGDfile)
        np.save("{}{}{}".format(result_path,saveFileName,"x_train"),x_train)
        np.save("{}{}{}".format(result_path,saveFileName, "y_train"), y_train)
        np.save("{}{}{}".format(result_path,saveFileName, "x_test"), x_test)
        np.save("{}{}{}".format(result_path, saveFileName, "y_test"), y_test)


# dataset="Aleax2LD,Aleax,yd_20180427"
def getDataFromFile(dataset):
    result_dir="../result_data/bi/"
    front="{}{}".format(result_dir, dataset)
    x_train=np.load("{}{}".format(front,"x_train.npy"))
    y_train=np.load("{}{}".format(front,"y_train.npy"))
    x_test=np.load("{}{}".format(front,"x_test.npy"))
    y_test=np.load("{}{}".format(front,"y_test.npy"))
    return (x_train, y_train), (x_test, y_test)



def biLSTM():

    batch_size = 32

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = getDataFromFile("Aleax2LD")
    print(y_train.shape)
    print("papre model...")
    model = Sequential()
    model.add(Embedding(20000, 32))
    model.add(Dense(16,activation='relu'))
    model.add(Bidirectional(LSTM(16)))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    # 尝试使用不同的优化器和优化器配置
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train, batch_size=batch_size,epochs=4)

    # print(model.evaluate(x_test,y_test))

    y_predict=model.predict(x_test)

    y_predict = (y_predict >= 0.5).astype(int)
    # print(y_test)
    # print(y_predict)
    print("acc={}".format(accuracy_score(y_test,y_predict)))
    print("precision={}".format(precision_score(y_test,y_predict)))
    print("recall={}".format(recall_score(y_test,y_predict)))


def cnn_lstm():
    # Embedding
    max_features = 20000
    maxlen = 100
    embedding_size = 32

    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4

    # LSTM
    lstm_output_size = 16

    # Training
    batch_size = 30
    epochs = 2

    '''
    注意:
    batch_size 是高度敏感的
    由于数据集非常小，因此仅需要 2 个轮次。
    '''

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = getDataFromFile("Aleax2LD")
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')

    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Bidirectional(LSTM(lstm_output_size)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    y_predict = model.predict(x_test)

    y_predict = (y_predict >= 0.5).astype(int)
    # print(y_test)
    # print(y_predict)
    print("acc={}".format(accuracy_score(y_test, y_predict)))
    print("precision={}".format(precision_score(y_test, y_predict)))
    print("recall={}".format(recall_score(y_test, y_predict)))


# def attention_biLSTM():
#     TIME_STEPS = 64
#     INPUT_DIM = 64
#     lstm_units = 64
#
#     # data pre-processing
#     # (X_train, y_train), (X_test, y_test) = mnist.load_data('mnist.npz')
#     # X_train = X_train.reshape(-1, 28, 28) / 255.
#     # X_test = X_test.reshape(-1, 28, 28) / 255.
#     # y_train = np_utils.to_categorical(y_train, num_classes=10)
#     # y_test = np_utils.to_categorical(y_test, num_classes=10)
#     # print('X_train shape:', X_train.shape)
#     # print('X_test shape:', X_test.shape)
#
#     (X_train, y_train), (X_test, y_test) = getDataFromFile("Aleax2LD")
#
#
#     # first way attention
#     def attention_3d_block(inputs):
#         # input_dim = int(inputs.shape[2])
#         # print(input_dim)
#         a = Permute((1, 1))(inputs)
#         a = Dense(TIME_STEPS, activation='softmax')(a)
#         a_probs = Permute((1, 1), name='attention_vec')(a)
#         # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
#         output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
#         return output_attention_mul
#
#     # build RNN model with attention
#     inputs = Input(shape=(64,))
#     drop1 = Dropout(0.3)(inputs)
#     lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(drop1)
#     attention_mul = attention_3d_block(lstm_out)
#     attention_flatten = Flatten()(attention_mul)
#     drop2 = Dropout(0.3)(attention_flatten)
#     output = Dense(1, activation='sigmoid')(drop2)
#     model = Model(inputs=inputs, outputs=output)
#
#     # second way attention
#     # inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
#     # units = 32
#     # activations = LSTM(units, return_sequences=True, name='lstm_layer')(inputs)
#     #
#     # attention = Dense(1, activation='tanh')(activations)
#     # attention = Flatten()(attention)
#     # attention = Activation('softmax')(attention)
#     # attention = RepeatVector(units)(attention)
#     # attention = Permute([2, 1], name='attention_vec')(attention)
#     # attention_mul = merge([activations, attention], mode='mul', name='attention_mul')
#     # out_attention_mul = Flatten()(attention_mul)
#     # output = Dense(10, activation='sigmoid')(out_attention_mul)
#     # model = Model(inputs=inputs, outputs=output)
#
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     print(model.summary())
#
#     print('Training------------')
#     model.fit(X_train, y_train, epochs=10, batch_size=16)
#
#     print('Testing--------------')
#     loss, accuracy = model.evaluate(X_test, y_test)
#
#     print('test loss:', loss)
#     print('test accuracy:', accuracy)

if __name__=="__main__":
    biLSTM()