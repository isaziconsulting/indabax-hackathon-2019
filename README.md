# Isazi IndabaX Hackathon

### Welcome to the Isazi Consulting Handwriting Recognition challenge at Deep Learning IndabaX South Africa 2019!

Follow the instructions below (in order) to get started.

If you have any questions about the challange, the dataset or model structure, ask one of the tutors.

If you run into any errors (e.g. out of memory or failing to load PyTorch), spend 10 minutes trying to figure it out. If you are still having trouble, ask a tutor.

## Installing Dependencies

### With Conda

The easiest way to get up and running quickly is with Anaconda/Miniconda (although it can sometimes interfere with other stuff in weird ways).

Install the Python 3.7 version from here: https://docs.conda.io/en/latest/miniconda.html

Then install PyTorch:
```
conda create -n indabax pytorch torchvision cudatoolkit=9.0 -c pytorch
source activate conda
```

This assumes you have an Nvidia GPU and CUDA 9.


### Without Conda

If you don't like Conda, it's not hard to install everything without it:

1.  Make sure you have Python 3.6 or later (`python3 -V`)
2.  `python3 -m venv isazi-indabax-venv`
3.  `. isazi-indabax-venv/bin/activate`
4.  `pip install numpy`
5.  `pip install torch torchvision`

This will work even if you don't have a GPU.

The last command will be slightly different if you have CUDA 8.


## Test your installation

To check that PyTorch is installed correctly, run `python src/test_pytorch.py`.

If that fails, first make sure you followed the instructions above.

If you did, call one of the tutors.


## Fetch the training data

1.  Make sure you are in the root of this repo (where this README file is)
2.  Download the data: `wget https://storage.googleapis.com/isazi-indabax-hackathon/phase_1.tar`
3.  `tar xaf phase_1.tar`

Have a look at some of the images in `phase_1/images`. You will see there is a variety of handwriting styles and weird backgrounds that your model will have to learn to handle.


## Run the training script

1.  `cd src`
2.  `python train_model.py`

It should start printing out info about the current training status every 100 batches (see the example below). If there is an error, call a tutor.

```
Train Epoch: 0 [800/14724 (5%)] Loss: 2.604970  Time per batch: 0.14 s  Total Time: 0.00 hrs
Train Epoch: 0 [1600/14724 (11%)]       Loss: 2.597058  Time per batch: 0.12 s  Total Time: 0.01 hrs
...
...
...
Train Epoch: 0 [14400/14724 (98%)]      Loss: 2.678329  Time per batch: 0.12 s  Total Time: 0.06 hrs
Avg. Epoch Loss: 2.793641

Test Prediction
['', '', '', '', '', '', '', '']
Test Ground Truth
['697161160', '20021112', '982346', '6844519', '031205', '022060', '000823196', '19651116'] 

Avg. Test Loss: 2.6005  CER: 100.0%     WER: 1841/1841 (100.0%) 
```


## Train a model

1.  Have a look at the code (most importantly the `Model` class). If you don't understand something, Google it, then ask a tutor.
2.  Modify the code in some clever way
3.  ???
4.  Profit

NOTE: the script only saves the model AFTER training has converged. You could modify the source to save it after every epoch, for example, so that you won't lose your progress if training stops for whatever reason.


## Test your model

Once you have trained your improved model, you need evaluate it using the validation set, and later the test set. You can do this by running the `inference.py` file and modifying the `PHASE` and `TEST_LIST` variables depending which dataset you want to use.

This will write a file called `submission.csv`. Open the file and make sure it has two colums (file ID and the models prediction). This is the file that you will submit to get on the leaderboard.
