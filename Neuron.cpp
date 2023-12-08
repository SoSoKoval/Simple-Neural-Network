#include "Neuron.h"

double Neuron::eta = 0.15; // learning rate
double Neuron::alpha = 0.5; // momentum

void Neuron::updateInputWeights(Layer &prevLayer){
    for(int i = 0; i < prevLayer.size(); ++i){
        Neuron &neuron = prevLayer[i];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const{
    double sum = 0.0;
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n){
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient; //sum of errors
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer){
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVals){
    double delta = targetVals - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x){
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x){
    return 1 /(cosh(x) * cosh(x));
}

void Neuron::feedForward(const Layer &prevLayer){
    double sum = 0.0;
    for(int i = 0 ; i < prevLayer.size(); ++i){
        sum += prevLayer[i].getOutputVal() * prevLayer[i].m_outputWeights[m_myIndex].weight;
    }
    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for(unsigned c = 0; c < numOutputs; ++c){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}