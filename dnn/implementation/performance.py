#!/usr/bin/env python3
import numpy as np
import torchaudio
import glob
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import hyperparameters

pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')

#TODO: 2 pesq's berechnen. 1.: clean und prediction 2.: noise und prediction. Dann ein Delta berechnen: 1.-2. Im Predict würde das dann aufgerufen werden.

data_clean = glob.glob(hyperparameters.OUT_DIR+"/clean*.wav")
print(data_clean)
#data_mix = np.sort(np.array(glob.glob(hyperparameters.OUT_DIR+"mix/*.wav")))
#data_pred = np.sort(np.array(glob.glob(hyperparameters.OUT_DIR+"pred/*.wav")))
dictionary = {}
for file_name in data_clean[:5]:
    clean_file, _ = torchaudio.load(file_name)
    mix_file, _ = torchaudio.load(file_name.replace('clean', 'mix'))
    pred_file, _ = torchaudio.load(file_name.replace('clean', 'pred'))
    noise_file = mix_file - clean_file

    pesq_pred_to_clean = pesq(pred_file[0], clean_file[0])
    pesq_pred_to_noise = pesq(pred_file[0], noise_file[0])
    pesq_delta = pesq_pred_to_clean - pesq_pred_to_noise

    #TODO: Abspeichern z.b. als dictionary inkl. Dateinamen

    dictionary[file_name] = pesq_delta.item()

print(dictionary)


