#!/usr/bin/env python3
import numpy as np
import torchaudio
import glob
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import hyperparameters

#TODO: SNR vs. PESQ, Entfernung der kürzesten Störquelle vs. PESQ, dafür Mittelwert vom PESQ mit Konfidenzintervall.
#TODO: SISDR, PESQ, Revibration Rate, Entfernung kürzeste Störquelle, SNR und natürlich File ID (nur clean? dann darauf achten, dass das auch konsistent bleibt) das alles mit pandas.dataFrame

def __calculate_pesq__():
    pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')

    data_clean = glob.glob(hyperparameters.OUT_DIR+"/clean*.wav")

    #print(data_clean)

    dictionary = {}
    for file_name in data_clean:
        clean_file, _ = torchaudio.load(file_name)
        mix_file, _ = torchaudio.load(file_name.replace('clean', 'mix'))
        #mix_file, _ = torchaudio.load(file_name.replace('clean', 'mixture'))
        pred_file, _ = torchaudio.load(file_name.replace('clean', 'pred'))
        #pred_file, _ = torchaudio.load(file_name.replace('clean', 'mixture'))
        noise_file = mix_file - clean_file
        if pred_file.sum() != 0 and clean_file.sum() != 0 and noise_file.sum() != 0:
            try:
                pesq_pred_to_clean = pesq(pred_file[0], clean_file[0])
                pesq_pred_to_noise = pesq(pred_file[0], noise_file[0])
                pesq_delta = pesq_pred_to_clean - pesq_pred_to_noise
            except:
                print('Error: 213')


            dictionary[file_name] = pesq_delta.item()

    print(dictionary)

if __name__ == "__main__":
    __calculate_pesq__()
