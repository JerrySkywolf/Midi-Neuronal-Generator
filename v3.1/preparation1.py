import numpy as np
from mido import Message, MidiFile
from random import randint

from os import listdir, environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.models import load_model


def create_midi_data(midi_path: str):
    database_list = []

    for file in listdir(path=midi_path):
        if '.mid' in file:
            mid = MidiFile(f'{midi_path}/{file}')
            for t, track in enumerate(mid.tracks):
                if t > 0:
                    for msg in track:
                        if type(msg) == Message:
                            msg = msg.copy()
                            if msg.type == 'note_on':
                                flag = 1
                            elif msg.type == 'note_off':
                                flag = 0
                            else:
                                continue
                            unit = (flag, msg.note, msg.velocity, msg.time)
                            database_list.append(unit)

    database_list += database_list[::-1]

    return database_list

def create_databases(midi_path: str, train_length: int, step: int):
    database_list = create_midi_data(midi_path=midi_path)
    x_list = []
    y_list = []

    for i in range(0, len(database_list) - train_length, step):
        x_list.append(database_list[i: i + train_length])
        y_list.append(database_list[i + train_length])

    x = np.zeros(shape=(len(x_list), train_length, 4))
    y = np.zeros(shape=(len(y_list), 4))

    for i, pair in enumerate(x_list):
        for j, unit in enumerate(pair):
            for k, note in enumerate(unit):
                x[i, j, k] = int(note) / 128
    for i, unit in enumerate(y_list):
        for j, note in enumerate(unit):
            y[i, j] = int(note) / 128

    validate_size = int(len(x_list) * 0.9)
    validate_x = x[validate_size:]
    validate_y = y[validate_size:]
    train_x = x[:validate_size]
    train_y = y[:validate_size]

    return train_x, train_y, validate_x, validate_y

def read_model(version: int):
    model = load_model(filepath=f'model_{version}.keras')
    return model

def get_seed():
    a, b, c, d = create_databases(midi_path='midi', train_length=8, step=1)
    e = np.zeros(shape=(1, 8, 4))
    split_position = randint(16, 256)
    for i in range(0, 8):
        e[0, :, :] = a[split_position, :, :]
    print(split_position)
    print(e)
    return e

if __name__ == '__main__':
    get_seed()

