from os import environ

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from time import time
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Input, GRU
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping, ModelCheckpoint

from preparation1 import create_databases


superparameters = {'batch_size': 256, 'train_length': 8}
structure = {'GRU1':20, 'GRU2': 16, 'Dense': 4, 'optimizer':'Adam', 'lr': 0.0156}
train_x, train_y, validate_x, validate_y = create_databases(midi_path='midi', train_length=superparameters['train_length'], step=1)

epochs = 1024
version = int(time())

model = Sequential([
    Input(shape=(superparameters['train_length'], structure['Dense']), batch_size=superparameters['batch_size']),
    GRU(units=structure['GRU1'], return_sequences=True),
    GRU(units=structure['GRU2'], return_sequences=False),
    Dense(units=structure['Dense'], activation='softmax'),
])

optimizer = Adam(learning_rate=structure['lr'])
model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=['accuracy'])
callbacks_list = [
EarlyStopping(monitor='val_accuracy', patience=128, mode='max'),
ModelCheckpoint(filepath=f'model_{version}_best.keras', monitor='val_accuracy', save_best_only=True)
]
model.summary()

history = model.fit(
                    x=train_x,
                    y=train_y,
                    batch_size=superparameters['batch_size'],
                    epochs=epochs,
                    callbacks=callbacks_list,
                    validation_data=(validate_x, validate_y)
                    )

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']


plt.plot(range(0, len(accuracy)), accuracy, 'b--',label='Train Accuracy')
plt.plot(range(0, len(val_accuracy)), val_accuracy, label='Validation Accuracy')
plt.suptitle(f'Effect of GRU Model:{version}')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(f'Parameters: {str(structure)}')
plt.legend()
plt.savefig(fname=f'Model_{version}_Analysis.png', dpi=1980)
plt.show()

model.save(filepath=f'Model_{version}_ep{len(val_accuracy)}_va{round(max(val_accuracy), 3)}.keras')
