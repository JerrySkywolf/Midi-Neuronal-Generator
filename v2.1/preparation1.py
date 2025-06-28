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
                            hex_mid = msg.hex()
                            hex_mid = list(hex_mid)
                            if len(hex_mid) > 6:
                                if hex_mid[0] == '8':
                                    hex_mid[0] = '0'
                                    unit = (hex_mid[0], hex_mid[1], hex_mid[3] + hex_mid[4], hex_mid[6] + hex_mid[7])
                                    database_list.append(unit)
                                if hex_mid[0] == '9':
                                    hex_mid[0] = '80'
                                    unit = (hex_mid[0], hex_mid[1], hex_mid[3] + hex_mid[4], hex_mid[6] + hex_mid[7])
                                    database_list.append(unit)
                                else:
                                    continue

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
                x[i, j, k] = int(note, 16) / 128
    for i, unit in enumerate(y_list):
        for j, note in enumerate(unit):
            y[i, j] = int(note, 16) / 128

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

