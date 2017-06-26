from time import time
import os
import warnings
import numpy as np
from random import shuffle
from skimage.data import imread
from skimage.transform import resize
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.regularizers import l1, l2
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score

t0 = time()
warnings.filterwarnings('ignore')
print('Loading Images')
image = []
path = 'classification_data'
n = 7  # dummy declaration
j = 0
for filename in os.listdir(path):
    img = imread(os.path.join(path, filename))
    img = resize(img, (48, 64))
    for i, s in enumerate(filename):
        if s is '_':
            n = int(filename[0:i])
    image.append([img, n])

shuffle(image)
labels = []
image1 = []
for i in range(len(image)):
    labels.append(image[i][1])
    image1.append(image[i][0])
label = LabelEncoder().fit_transform(labels)
t1 = time()
T = t1-t0
print('Time to load the images  :   %0.2f sec' % T)
print('Split into train and test batches images')
image1 = np.asarray(image1)
X_train, X_test, y_train, y_test = train_test_split(image1, label, test_size=0.2)

t2 = time()
T = t2-t1
print('Time take to split images  :  %0.2f sec' % T)


def classifier_fun():
    print('Load the classifier function and build the model')
    clf = Sequential()
    clf.add(Conv2D(32, (3, 3), input_shape=(96, 128, 3), activation='relu'))
    clf.add(MaxPooling2D(pool_size=(2, 2)))
    clf.add(Conv2D(64, (3, 3), activation='relu'))
    clf.add(MaxPooling2D(pool_size=(2, 2)))
    clf.add(Conv2D(128, (3, 3), activation='relu'))
    clf.add(MaxPooling2D(pool_size=(2, 2)))
    clf.add(Flatten())
    clf.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01),
                  bias_initializer='glorot_uniform'))
    clf.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01),
                  bias_initializer='glorot_uniform'))
    clf.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01),
                  bias_initializer='glorot_uniform'))
    clf.add(Dense(5, activation='softmax', kernel_regularizer=l2(0.01),
                  bias_initializer='glorot_uniform'))
    opt = adam(lr=0.01, decay=0.00001)
    clf.compile(optimizer=opt, loss='categorical_crossentropy',
                metrics=['categorical_accuracy'])
    clf.summary()

    return clf

print('Call the classifier function')
model = classifier_fun()
t3 = time()
T = t3 - t2
print('\nTime to load the build the classifier  :  %0.2f sec' % T)

train_image_data_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                          zca_whitening=True, horizontal_flip=True, vertical_flip=True,
                                          rotation_range=45, width_shift_range=0.25, height_shift_range=0.25,
                                          shear_range=0.25, zoom_range=0.25, channel_shift_range=0.25,
                                          fill_mode='reflect', rescale=1./255)
test_image_data_gen = ImageDataGenerator(rescale=1./255)

fold = KFold(n_splits=5)
ypredv = np.empty((1, 5))
ytruev = np.empty((1, 5))
fscore = []
accuracy = []
prs = []
roc = []
c = 0
for train, valid in fold.split(X_train, y_train):
    c += 1

    print('\nValidation Loop : ', c)
    print('\nSplit into training and validation data')
    Xtrain = X_train[train]
    ytrain = y_train[train]
    Xvalid = X_train[valid]
    yvalid = y_train[valid]
    print('One hot encoded labels')
    ytrain_label = to_categorical(ytrain, 5)
    yvalid_label = to_categorical(yvalid, 5)
    print('Fit train data to image generator')   # taking too much time
    t01 = time()
    train_image_data_gen.fit(Xtrain)
    t02 = time()
    t = t02 - t01
    print('Time taken to fit data to Image gen  :  %0.2f sec' % t)
    train_set = train_image_data_gen.flow(Xtrain, ytrain_label, batch_size=32)
    test_set = test_image_data_gen.flow(Xvalid, yvalid_label, batch_size=32)
    print('fit the data to classifier')
    t03 = time()
    model.fit_generator(train_set, steps_per_epoch=len(Xtrain)/32,
                        validation_data=test_set, validation_steps=len(Xvalid)/32)
    t04 = time()
    t = t04 - t03
    print('Time taken to fit data to classifier  :  %0.2f sec' % t)
    print('predict from validation data')
    y = model.predict_generator(test_set, steps=len(Xvalid)/32)
    prs.append(average_precision_score(y_true=yvalid_label, y_score=y, average='weighted'))
    roc.append(roc_auc_score(y_true=yvalid_label, y_score=y, average='weighted'))
    if c == 1:
        ytruev = yvalid_label
        ypredv = y
    else:
        ytruev = np.append(ytruev, yvalid_label, axis=0)
        ypredv = np.append(ypredv, y, axis=0)
    for i in range(len(y)):
        for k in range(len(y[i])):
            if y[i][k] >= 0.65:
                y[i][k] = 1
            else:
                y[i][k] = 0
    fscore.append(f1_score(y_true=yvalid_label, y_pred=y, average='weighted'))
    accuracy.append(accuracy_score(y_true=yvalid_label, y_pred=y))

model.save('cnn_model.h5')

t4 = time()
T = t4 - t3
print('\n\n\n\n\nTime to perform the validation  :  %0.2f sec' % T)
print('\n Total CV Area under PR curve  =  ',
      average_precision_score(y_true=ytruev, y_score=ypredv, average='weighted'))
print('Total CV Area under ROC curve  =  ',
      roc_auc_score(y_true=ytruev, y_score=ypredv, average='weighted'))
print('Mean CV Area under PR curve  =  ', np.mean(prs))
print('Mean CV Area under ROC curve  =  ', np.mean(roc))
print('Mean CV f1 score  =  ', np.mean(fscore))
print('Mean CV Accuracy  =  ', np.mean(accuracy))
print('\n Prediction of Test data')
ytest_label = to_categorical(y_test, 5)
yp = model.predict_generator(test_image_data_gen.flow(X_test, y_test, batch_size=32),
                             steps=len(X_test)/32)
T = time()-t4
print('Time for prediction  :  %0.2f sec' % T)
print('\n Area under PR curve  =  ',
      average_precision_score(y_true=ytest_label, y_score=yp, average='weighted'))
print('\n Area under ROC curve  =  ',
      roc_auc_score(y_true=ytest_label, y_score=yp, average='weighted'))
for i in range(len(yp)):
    for k in range(len(yp[i])):
        if yp[i][k] >= 0.65:
            yp[i][k] = 1
        else:
            yp[i][k] = 0
print('\n f1-score  =  ', f1_score(y_true=ytest_label, y_pred=yp, average='weighted'))
print('\n Accuracy  =  ', accuracy_score(y_true=ytest_label, y_pred=yp))

T = time() - t0
print('\n Time taken for the code to run is --->  %0.2f sec\n\n\n' % T)
print('\n\n\n-x-x-x-x-x-THE END-x-x-x-x-x-')
