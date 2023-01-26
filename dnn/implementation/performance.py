#!/usr/bin/env python3
import numpy as np
import torchaudio
import glob
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import hyperparameters
import pandas as pd
import json


#TODO: SNR vs. PESQ, Entfernung der kürzesten Störquelle vs. PESQ, dafür Mittelwert vom PESQ mit Konfidenzintervall.
#TODO: SISDR, PESQ, Revibration Rate, Entfernung kürzeste Störquelle, SNR und natürlich File ID (nur clean? dann darauf achten, dass das auch konsistent bleibt) das alles mit pandas.dataFrame

def __calculate_pesq__():
    pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')

    data_clean = np.sort(np.array(glob.glob(hyperparameters.OUT_DIR+"/clean*.wav")))
    data_meta = np.sort(np.array(glob.glob(hyperparameters.OUT_DIR+"/meta*")))

    #print(data_clean)

    dictionary = {'data_index': [] , 'SNR':[], 'SISDR':[], 'PESQ':[], 'Reverberation':[], 'DisToNoise':[]}
    index = 0
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
                pesq_delta = (pesq_pred_to_clean - pesq_pred_to_noise).item()
            except:
                print('Error: 213')


            dictionary['PESQ'].append(pesq_delta) 
        #Read Meta Data
        meta_file = open(data_meta[index])
        meta_json = json.loads(meta_file.read())
        dictionary['data_index'].append(meta_json["data_index"]) 
        dictionary['Reverberation'].append(meta_json["reverberation_rate"])
        dictionary['SNR'].append(meta_json["SNR"]) 
        dictionary['SISDR'].append(meta_json["SISDR"]) 
        dictionary['DisToNoise'].append(meta_json["min_distance_to_noise"]) 
        
        index += 1
            
    Data_Matrix = pd.DataFrame(data=dictionary)

    Data_Matrix.to_csv(hyperparameters.OUT_DIR+'/Dataframe.csv')

if __name__ == "__main__":
    __calculate_pesq__()
