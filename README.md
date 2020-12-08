# Using Neural Networks to Develop an AI Capable of Learning to Effectively Play Video Games

Brian Crutchley                 bcrutc01@rams.shepherd.edu

Grayson Grzadzielewski          ggrzado1@rams.shepherd.edu

## Installation

To install the requirements for this repo, either use the requirements.txt file to download dependencies, or do

```
$ pip install .
```

## Usage

### NEAT Implementation

To run the implementation of NEAT method, do

```
$ python main.py --neat
```

To test the most recent saved generation of the NEAT implementation

```
$ python main.py --neat --test
```

### Curiosity Implementation

To run the implementation of the Curiosity method, do

```
$ python main.py --curiosity --name <name of save file>
```

To test a saved model of the Curiosity implementation

```
$ python main.py --curiosity --name <name of saved model> --test
```

#### Note on Curiosity Memory Requirements

The Curiosity implementation requires a significant amount of memory to run. By default, the trajectory size may be too large. To train using a different trajectory size, do

```
$ python main.py --curiosity --name <name of save file> --trajectory <int trajectory size>
```

To train the model using the best 25% of the trajectory as a batch, do

```
$ python main.py --curiosity --name <name of save file> --trajectory <int trajectory size> --exp
```
