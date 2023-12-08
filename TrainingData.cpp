#include "TrainingData.h"

void TrainingData::getTopology(vector<int> &topology){
    string line;
    getline(m_trainingDataFile, line);
    stringstream stream(line);
    string label;
    stream >> label;
    if(this->isEof() || label.compare("topology:") != 0){
        abort();
    }
    while(!stream.eof()){
        int cur;
        stream >> cur;
        topology.push_back(cur);
    }
}

TrainingData::TrainingData(const string filename){
    m_trainingDataFile.open(filename.c_str());
}

int TrainingData::getNextInputs(vector<double> &inputVals){
    inputVals.clear();
    string line;
    getline(m_trainingDataFile, line);
    stringstream stream(line);
    string label;
    stream >> label;
    if (label.compare("in:") == 0) {
        double curInputValue;
        while (stream >> curInputValue) {
            inputVals.push_back(curInputValue);
        }
    }
    return inputVals.size();
}

int TrainingData::getTargetOutputs(vector<double> &targetOutputVals){
    targetOutputVals.clear();
    string line;
    getline(m_trainingDataFile, line);
    stringstream stream(line);
    string label;
    stream >> label;
    if (label.compare("out:") == 0) {
        double curTargetValue;
        while (stream >> curTargetValue) {
            targetOutputVals.push_back(curTargetValue);
        }
    }
    return targetOutputVals.size();
}
