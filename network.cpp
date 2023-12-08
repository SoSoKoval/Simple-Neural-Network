#include "network.h"

double network::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

void network::getResults(vector<double> &resultVals) const{
    resultVals.clear();
    for(int i = 0; i < m_layers.back().size() - 1; ++i){
        resultVals.push_back(m_layers.back()[i].getOutputVal());
    }
}

void network::backProp(const std::vector<double> &targetVals){
    // RMS of output neuron errors
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // average error squared
    m_error = sqrt(m_error); // RMS
    // Implement a recent average measurement:
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);
    // output layer gradients
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }
    // gradients on hidden layers
    for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum){
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];
        for(unsigned n = 0; n < hiddenLayer.size(); ++n){
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }
    // update connection weights for all layers
    for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum){
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];
        for(unsigned n = 0; n < layer.size() - 1; ++n){
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void network::feedForward(const vector<double> &inputVals){
    assert(inputVals.size() == m_layers[0].size() - 1);
    for(int i = 0; i < inputVals.size(); ++i){
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    for(int layerNum = 1; layerNum < m_layers.size(); ++layerNum){
        Layer &prevLayer = m_layers[layerNum - 1];
        for(int i = 0; i < m_layers[layerNum].size() - 1; ++i){
            m_layers[layerNum][i].feedForward(prevLayer);
        }
    }
}

network::network(const vector<int> &topology){
    int numLayers = topology.size();
    for(int layerNum = 0; layerNum < numLayers; ++layerNum){
        m_layers.push_back(Layer());
        int numOutputs = layerNum == topology.size() - 1 ? 0 :topology[layerNum + 1];
        for(int neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Made a Neuron!" << endl;
        }
        m_layers.back().back().setOutputVal(1.0);
    }
}