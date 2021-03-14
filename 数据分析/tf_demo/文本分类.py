# -*-coding:utf-8-*-
"""
Created on 2019/9/23 17:20
@author: joe
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# print("Training entire:{}, labels:{}".format(len(train_data),len(train_labels)))

word_index = imdb.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["UNUSED"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(train_data[0]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(20, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
results = model.evaluate(test_data, test_labels)
y_pre = model.predict(test_data)
print(y_pre)
print(results)
#
# acc = history.history.get('acc')
# val_acc = history.history.get('val_acc')
# loss = history.history.get('loss')
# val_loss = history.history.get('val_loss')
#
# # "bo" is for "blue dot"
# plt.plot(history.epoch, loss, 'bo', label='Training loss')
# # b is for "solid blue line"
# plt.plot(history.epoch, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
#
# plt.clf()  # clear figure
# acc_values = history.history.get('acc')
# val_acc_values = history.history.get('val_acc')
#
# plt.plot(history.epoch, acc_values, 'bo', label='Training acc')
# plt.plot(history.epoch, val_acc_values, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.show()
