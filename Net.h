#pragma once
#include <vector>

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

class Net{
public:
	Net(const vector<unsigned> topology);
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }
	vector<Layer> m_layers;// m_layers[layerNum][neuronNum]
	void saveWeight(Net myNet, string fileName);
	void loadWeights(Net myNet,vector<vector<string> > weightArray, string fileName);
	
private:
	double m_error;
	
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};
