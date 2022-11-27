#!/usr/bin/env python3
import numpy as np
import torchaudio
import glob
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import hyperparameters

def __calculate_pesq__():
    pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')

    #TODO: 2 pesq's berechnen. 1.: clean und prediction 2.: noise und prediction. Dann ein Delta berechnen: 1.-2. Im Predict würde das dann aufgerufen werden.

    data_clean = glob.glob(hyperparameters.OUT_DIR+"/clean*.wav")

    # print(data_clean)

    dictionary = {}
    for file_name in data_clean:
        clean_file, _ = torchaudio.load(file_name)
        mix_file, _ = torchaudio.load(file_name.replace('clean', 'mix'))
        #mix_file, _ = torchaudio.load(file_name.replace('clean', 'mixture'))
        pred_file, _ = torchaudio.load(file_name.replace('clean', 'pred'))
        #pred_file, _ = torchaudio.load(file_name.replace('clean', 'mixture'))
        noise_file = mix_file - clean_file
        if pred_file.sum().data[0] != 0 and clean_file.sum().data[0] != 0 and noise_file.sum().data[0] != 0:
            pesq_pred_to_clean = pesq(pred_file[0], clean_file[0])
            pesq_pred_to_noise = pesq(pred_file[0], noise_file[0])
            pesq_delta = pesq_pred_to_clean - pesq_pred_to_noise

        #Abspeichern als dictionary inkl. Dateinamen

            dictionary[file_name] = pesq_delta.item()

            print(dictionary)


