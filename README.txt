Using Neural Networks to Develop an AI Capable of Learning to Effectively Play Video Games

Brian Crutchley bcrutc01@rams.shepherd.edu

Grayson Grzadzielewski ggrzad01@rams.shepherd.edu
Installation
PyTorch

Because PyTorch varies by OS, specific commands are required to download it for your system. If none of the commands below work, see https://pytorch.org/get-started/locally/

For Windows OS

$ pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

For Linux OS

$ pip install torch torchvision

For Mac OS

$ pip install torch torchvision torchaudio

Remaining Python Packages

There are two options to install the remaining python dependencies required for this repo to function. The first is to use the following commands to install from requirements.txt and move the integration files into your retro/data/src/stable/SuperMarioBros-Nes directory.

$ pip install -r requirements.txt
$ python integration/patch_integration.py

Or simply install with setup.py

$ python setup.py install

Acquiring Game ROMs

NOTE: Below it says that the ROMs to run are not included in the repository! The correct ROM is included in the zip file we're turning in

While this repo does not distribute the ROMs required for running, they are available here: https://archive.org/details/No-Intro-Collection_2016-01-03_Fixed. To install the ROMs to the retro/data/stable directory

    1. Download the archive and unzip

    2. cd into the archive’s directory

      * The ROM files have a .zip extension. DO NOT UNZIP THE ROM files INSIDE the archive!

    3. Do the following command. Do not exclude the "."

$ python -m retro.import .

Usage
NEAT Implementation

To run the implementation of NEAT method, do

$ python main.py --neat

To test the most recent saved generation of the NEAT implementation

$ python main.py --neat --test

Curiosity Implementation

To run the implementation of the Curiosity method, do

$ python main.py --curiosity --name <name of save file>

To test a saved model of the Curiosity implementation

$ python main.py --curiosity --name <name of saved model> --test

Note on Curiosity Memory Requirements

The Curiosity implementation requires a significant amount of memory to run. By default, the trajectory size may be too large. To train using a different trajectory size, do

$ python main.py --curiosity --name <name of save file> --trajectory <int trajectory size>

To train the model using the best 25% of the trajectory as a batch, do

$ python main.py --curiosity --name <name of save file> --trajectory <int trajectory size> --exp
