#include "TrainingData.h"
#include "network.h"

void printV(string name, vector<double> &v){
    cout << name << ' ';
    for(unsigned i = 0; i < v.size(); ++i){
        cout << abs(v[i]) << " ";
    }
    cout << endl;
}

int main(){
    TrainingData trainData("C:\\Users\\Zhora_loh\\CLionProjects\\network\\trainingData.txt"); // ented here the path to the document with data
    vector<int> topology;
    trainData.getTopology(topology);
    network myNet(topology);
    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;
    while(!trainData.isEof()){
        trainingPass++;
        cout << '\n' << "Pass" << trainingPass;
        if(trainData.getNextInputs(inputVals) != topology[0])
            break;
        printV(": Input:", inputVals);
        myNet.feedForward(inputVals);
        myNet.getResults(resultVals);
        printV("Output:", resultVals);
        trainData.getTargetOutputs(targetVals);
        printV("Targets:", targetVals);
        assert(targetVals.size() == topology.back());
        myNet.backProp(targetVals);
        cout << "Net recent average error: " << myNet.getRecentAverageError() << '\n';
    }
    cout << '\n' << "Done" << '\n';
    return 0;
}