cmake_minimum_required(VERSION 3.25)
project(network)

set(CMAKE_CXX_STANDARD 17)

add_executable(network main.cpp network.cpp network.h Neuron.cpp Neuron.h TrainingData.cpp TrainingData.h)
