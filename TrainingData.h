#ifndef NETWORK_TRAININGDATA_H
#define NETWORK_TRAININGDATA_H
#include <bits/stdc++.h>
using namespace std;

class TrainingData{
public:
    TrainingData(const string filename);
    bool isEof(void){ return m_trainingDataFile.eof(); }
    void getTopology(vector<int> &topology);
    int getNextInputs(vector<double> &inputVals);
    int getTargetOutputs(vector<double> &targetOutputVals);
private:
    ifstream m_trainingDataFile;
};

#endif