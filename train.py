import h5py
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.preprocessing.image import *

if __name__ == '__main__':
    np.random.seed(2017)

    X_train = []
    # X_test = []

    for filename in ["gap_ResNet50.h5", "gap_InceptionV3.h5", "gap_Xception.h5"]:
        print('加载'+filename)
        with h5py.File(filename, 'r') as h:
            X_train.append(np.array(h['train']))
            # X_test.append(np.array(h['test']))
            y_train = np.array(h['label'])

    X = np.concatenate(X_train, axis=1)
    # X_test = np.concatenate(X_test, axis=1)

    X_train, y_train = shuffle(X, y_train)

    input_tensor = Input(X_train.shape[1:])
    x = input_tensor
    x = Dropout(0.5)(x)
    x = Dense(100, activation='softmax')(x)
    model = Model(input_tensor, x)

    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    y_train = np_utils.to_categorical(y_train, 100)

    model.fit(X_train, y_train, batch_size=64, nb_epoch=100, validation_split=0.1)

    # model.save('model.h5')

    y_pred = model.predict(X_train, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred[0])
    # test_filenames = []
    # with open('filename.txt', 'r') as f:
    #     for line in f:
    #         test_filenames.append(line.replace('\n', ''))
    #
    # with open('out/0717.txt', 'w') as f:
    #     for i, fname in enumerate(test_filenames):
    #         f.write(str(y_pred[i])+'\t'+fname+'\n')
