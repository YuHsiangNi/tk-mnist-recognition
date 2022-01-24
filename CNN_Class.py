import tensorflow as tf
from keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical

# 定義圖像寬、高
img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)

# 取得 MNIST 資料
def getData():
    # 載入 MNIST 訓練資料
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # y值轉成 one-hot encoding , 定義分類數量 = 10
    y_train = to_categorical(y_train, num_classes = 10)
    y_test = to_categorical(y_test, num_classes = 10)

    # x值 CNN Input 需要多一個 dimension
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    return x_train/255, y_train, x_test/255, y_test

def trainModel(x_train, y_train, x_test, y_test):
    batch_size = 64
    epochs = 12

    # 建立簡單的線性執行的模型
    model = tf.keras.models.Sequential()

    # 建立卷積層，filter=32,即 output space 的深度, Kernel Size: 3x3, activation function 採用 relu
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

    # 建立池化層，池化大小=2x2，取最大值
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.25
    model.add(Dropout(rate=0.25))

    # Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
    model.add(Flatten())

    # 全連接層: 128個output
    model.add(Dense(units=128, activation='relu'))

    # Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.5
    model.add(Dropout(0.5))

    # 使用 softmax activation function，將結果分類
    model.add(Dense(units=10, activation='softmax'))

    # 影像資料增強 Data Augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range = 10,
        shear_range = 0.1,
        zoom_range = 0.2,
        width_shift_range = 0.1,
        height_shift_range = 0.1
    )

    datagen.fit(x_train)

    # Define callbacks and learning rate
    csv_logger = tf.keras.callbacks.CSVLogger('training.log', separator = ',', append = False)
    earlyStop = tf.keras.callbacks.EarlyStopping(patience = 5)
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss',
        patience = 2,
        verbose = 1,
        factor = 0.1,
        min_lr = 0.00001
    )
    callbacks = [csv_logger, learning_rate_reduction, earlyStop]

    # 編譯: 選擇損失函數、優化方法及成效衡量方式
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    # 進行訓練, 訓練過程會存在 history 變數中
    history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                        validation_data=datagen.flow(x_test, y_test, batch_size=batch_size), verbose=1,
                        steps_per_epoch=x_train.shape[0]//batch_size, callbacks=callbacks)

    model.save('mnist_model.h5')

    return model

# 載入模型
def loadModel():
    return tf.keras.models.load_model('mnist_model.h5')



