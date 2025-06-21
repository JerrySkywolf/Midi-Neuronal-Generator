import numpy as np
from time import time
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from construction import read_model, create_notes, read_dictionary, encode, decode, make_path
from distribution import RandomDistributionReform


class Predict:
    def __init__(self,
                 epoch: int,
                 original_seed_file: str = 'seed.txt',
                 train_note_path: str = 'music txt',
                 result_save_path: str = 'results',
                 dictionary_path:str = 'dictionaries',
                 model_version=1749915232
                 ):
        path_list = [original_seed_file, train_note_path, result_save_path, dictionary_path]
        for path in path_list:
            make_path(dictionary_name=path)
        self.original_seed_file = original_seed_file
        self.save_path = result_save_path
        self.dictionary_path = dictionary_path
        self.seed_txt0 = self.make_seed0()
        self.notes = create_notes(path=train_note_path)
        self.note_indices, self.indices_note = read_dictionary(self.dictionary_path)
        self.model = read_model(version=model_version)
        self.prediction = None
        with open(file=f'{self.save_path}/result_{int(time())}.txt', mode='a+') as r:
            r.write(f'{self.seed_txt0} ')
            with open(file=self.original_seed_file, mode='a') as f:
                for i in range(0, epoch):
                    seed_list = self.make_seed(file=r)
                    #print(seed_list)
                    seed = encode(notes=seed_list, dictionary=self.note_indices, is_array=True)
                    self.prediction = self.model.predict(seed)
                    self.prediction = RandomDistributionReform(self.prediction, args=(0.8, 0.8))
                    self.prediction = self.prediction.beta_distribution()
                    mean = np.mean(self.prediction, axis=1)
                    max_indices = np.argmax(mean, axis=0)
                    predict_notes = decode(softmax_prediction=self.prediction, dictionary=self.indices_note, return_list=True)
                    predict_notes = predict_notes[int(max_indices)]
                    predict_notes_str = f'{predict_notes} '
                    '''
                    OR:
                    for l in predict_notes:
                        predict_notes_str += f'{l} '  
                    '''
                    r.write(predict_notes_str)
                    f.write(predict_notes_str)


    def make_seed0(self):
        seed_txt = ''
        with open(file=self.original_seed_file, mode='r') as f1:
            seed_txt += f1.read()
        return seed_txt

    @staticmethod
    def make_seed(file):
        seed_txt = ''
        file.seek(0)
        seed_txt += file.read()
        seed_list = seed_txt.split()
        seed_list = seed_list[-4:]
        return seed_list


if __name__ == '__main__':
    Predict(epoch=64)
