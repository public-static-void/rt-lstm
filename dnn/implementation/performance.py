#!/usr/bin/env python3
import numpy as np
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import torch
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')

stoi = ShortTimeObjectiveIntelligibility(8000, False)


