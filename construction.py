import numpy as np
from os import listdir, environ, mkdir
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.models import load_model


def make_path(dictionary_name: str):
    if dictionary_name not in listdir():
        mkdir(path=dictionary_name)

def encode(notes: list, dictionary: dict, max_length: int = 4, is_array=False):
    output = np.zeros(shape=(len(notes), max_length, len(dictionary)), dtype=np.bool)
    for i in range(0, len(notes)):
        for j in range(0, max_length):
            output[i, j, dictionary[notes[i]]] = 1
    if is_array:
        output = np.asarray(output)
    return output

def decode(softmax_prediction, dictionary: dict, return_list=True):
    predict_notes = np.zeros_like(softmax_prediction, dtype=np.bool)
    next_note_indices = []
    for i in range(0, softmax_prediction.shape[0]):
        next_note_position = np.argmax(softmax_prediction[i], axis=0)
        predict_notes[i, next_note_position] = 1
        next_note_indices.append(int(next_note_position))
    if return_list:
        next_notes = []
        for j in next_note_indices:
            next_notes.append(dictionary[j])
        predict_notes = next_notes
    return predict_notes

def read_model(version: int):
    make_path(dictionary_name='model_saves')
    model = load_model(filepath=f'model_saves/model_{version}.keras')
    return model

def read_dictionary(path: str = 'dictionaries'):
    make_path(dictionary_name=path)
    dictionary_list = []
    for file_name in listdir(path=f'{path}/'):
        if '.dictionary' in file_name:
            with open(file=f'{path}/{file_name}', mode='r') as f:
                dictionary_list.append(eval(f.read()))
    return tuple(dictionary_list)
