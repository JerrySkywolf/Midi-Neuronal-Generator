import numpy as np
from time import time
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from construction import read_model, read_dictionary, encode, decode, make_path
from distribution import RandomDistributionReform


class Predict:
    def __init__(self,
                 epoch: int,
                 original_seed_file: str = 'seed.txt',
                 result_save_path: str = 'results',
                 dictionary_path: str = 'dictionaries',
                 model_version: int = 1749915232
                 ):
        """
        Here is the core class of the program, which could be called to generate a sequence of notes, or music.
        This program is based on the license of 

        :param epoch: how many turns should the model predict, deciding the length of the generated music.
        :param original_seed_file: the name of a '.txt' file stored the Initial Seed.
        :param result_save_path: the directory where the result will be saved.
        :param dictionary_path: the directory where the notes-indices dictionaries is stored.
        :param model_version: the version of the model it used, which is actually the time when the model is trained.

        - 1.How to call this class?
        - Example:
            Predict(epoch=64, model_version=1749915232)
            # This will generate a piece of music with 64 notes.

        - 2.Where could I find the result?
        _ The generated music has been saved in the directory named 'result'.

        - 3.How to comprehend the result? It is only a sequence of characters.
        - The first part means the Note Name, and the second part means Duration, desperate with a '-', for example:
            <Note> 'C5-1/4' --> note C5, lasting for a quanter beat.
        """

        path_list = [original_seed_file, result_save_path, dictionary_path]
        for path in path_list:
            make_path(dictionary_name=path)

        self.original_seed_file = original_seed_file
        self.save_path = result_save_path
        self.dictionary_path = dictionary_path
        self.seed_txt0 = self.make_seed0()
        self.note_indices, self.indices_note = read_dictionary(self.dictionary_path)
        self.model = read_model(version=model_version)
        self.prediction = None

        with open(file=f'{self.save_path}/result_{int(time())}.txt', mode='a+') as r:
            r.write(f'{self.seed_txt0} ')

            for i in range(0, epoch):
                seed_list = self.make_seed(file=r)
                seed = encode(notes=seed_list, dictionary=self.note_indices, is_array=True)
                self.prediction = self.model.predict(seed)

                self.prediction = RandomDistributionReform(self.prediction, args=(0.8, 0.8))
                '''
                Parameters of the random distribution series can be adjusted here in the parameter 'args'.
                For instance: 
                    self.prediction = RandomDistributionReform(self.prediction, args=(0.5, 0.5))
                '''
                self.prediction = self.prediction.beta_distribution()
                '''
                The type of the random distribution series can be change here.
                Up to now, the beta distribution and the gamma distribution have been accessible.
                For instance:
                    self.prediction = self.prediction.gamma_distribution()
                '''

                mean = np.mean(self.prediction, axis=1)
                max_indices = np.argmax(mean, axis=0)
                predict_notes = decode(softmax_prediction=self.prediction, dictionary=self.indices_note, return_list=True)
                predict_notes = predict_notes[int(max_indices)]
                predict_notes_str = f'{predict_notes} '
                r.write(predict_notes_str)

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
