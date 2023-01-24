# LSTM For Real-Time Multichannel Speech Enhancement

## Repo Structure

- `net.py` defines the model itself and the dataloaders,

- `dataset.py` defines the processing of the data provided to the dataloaders for training and prediction,

- `hyperparameters.py` defines parameters used by the model and the runnables,

- the runnable scripts `train.py` and `predict.py` provide easy access to training and prediction functionality,

- while `rtpro.py` provides an interactive way of accessing the prediction functionality in a real-time environment.

## Requirements

First of all, make sure to set up a virtual environment and install all dependencies provided in the `requirements.txt` by running

    pip install -r requirements.txt

## How To Prepare The Data For Network Training

The data generation and preprocessing is outsourced into a separate branch `DatageneratorForLSTM`. Have a look there and follow the required steps to generate data before any network training can be performed.

Then, once data is generated, make sure it is located in correct directories in order to be accessed. By default it is assumed to be in `./soundfiles/generatedDatasets/` and then, depending on the purpose of the particular dataset (training, validation or prediction), followed by `Training`, `Validation` or `Test` respectively, resulting in

    ./soundfiles/generatedDatasets/Training
    ./soundfiles/generatedDatasets/Validation
    ./soundfiles/generatedDatasets/Test

The root directory path (`./soundfiles/`) can be adjusted by changing the `DATA_DIR` variable in `hyperparameters.py`.

## How To Use Net For Training Or Prediction

To train net run:

    python train.py

Checkpoints will be created in `./logs/lightning_logs` for the experiment specified by `v_num` and there in the subfolder `checkpoints`.

Before the net can be used for prediction, make sure you have a pretrained model checkpoint and the `trained_model_path` variable in `hyperparameters.py` set accordingly (to point to the location of the desired checkpoint). The checkpoint must resamble the same parameters as the net was trained on, especially the bidirectionality setting.

To use pretrained net for prediction run:

    python predict.py

The prediction script will write `wav`-soundfiles to the `./out` directory that can further be analyzed or just listened to.

There are plenty of adjustable settings in `hyperparameters.py`. Some of the probably more interesting might be:

- STFT shift: e.g. `stft_shift = 256` or `stft_shift = 128`
- Time-dimension LSTM bidirectionality: `t_bidirectional = True` or `t_bidirectional = False`
- Frequency-dimension LSTM bidirectionality: `f_bidirectional = True` or `f_bidirectional = False`

## How To Use Real-Time Processing

First, make sure to start up a `jack`-server with the current settings:

- samplerate of 48000 Hz
- 384 frames
- 4 buffer periods

Also, make sure the `batch_size`-parameter in `hyperparameters.py` is set to 1:

    batch_size = 1

Now, once `jack` is set up, in order to use the real-time processing script run:

    python rtpro.py
