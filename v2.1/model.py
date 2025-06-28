from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from time import time

from keras.models import Sequential
from keras.layers import Activation, Dense, Input, LSTM
from keras.optimizers import RMSprop
from keras.losses import CategoricalCrossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint

from preparation1 import create_databases


batch_size = 128
train_length = 8
train_x, train_y, validate_x, validate_y = create_databases(midi_path='midi', train_length=train_length, step=1)

model = Sequential([
    Input(shape=(train_length, 4), batch_size=batch_size),
    LSTM(units=128, return_sequences=False),
    Dense(units=4, activation='sigmoid'),
    Activation(activation='softmax')
])
optimizer = RMSprop(learning_rate=0.035)
model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])
callbacks_list = [
EarlyStopping(monitor='val_accuracy', patience=3, mode='max'),
ModelCheckpoint(filepath=f'model_{int(time())}.keras', monitor='val_accuracy', save_best_only=True)
]
model.summary()
model.fit(
    x=train_x,
    y=train_y,
    batch_size=batch_size,
    epochs=16,
    callbacks=callbacks_list,
    validation_data=(validate_x, validate_y)
)
