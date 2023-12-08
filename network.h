#ifndef NETWORK_NETWORK_H
#define NETWORK_NETWORK_H

#include "Neuron.h"

class network{
public:
    network(const vector<int> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }
private:
    vector<Layer> m_layers; //[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};

#endif
