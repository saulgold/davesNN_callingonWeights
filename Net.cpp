#include "Net.h"
#include "Neuron.h"
#include<vector>
#include<iostream>
#include<stdlib.h>
#include<cassert>
#include<cmath>
#include<fstream>
#include<sstream>
using namespace std;

void Net::loadWeights(Net myNet, vector<vector<string> > weightArray, string fileName){
	string line,value;
	ifstream inputWeights(fileName);
	unsigned row=0, col=0;
	

	while (!inputWeights.eof()){
		getline(inputWeights, line);
		istringstream iss(line);
		vector<string> newColumn;
		weightArray.push_back(newColumn);

		while (!iss.eof()){
			getline(iss, value, ',');
			
			weightArray.at(row).push_back(value);
			++col;
		}
		++row;
	}
	//print array
	for (unsigned row = 0; row < weightArray.size(); ++row){
		for (unsigned col = 0; col < weightArray[row].size(); ++col){
			cout << weightArray[row][col] << ", ";
		}
		cout << endl;
	}
	unsigned rowWeight = 0;
	for (unsigned layerNum = 0; layerNum< myNet.m_layers.size()-2; ++layerNum){
		Layer &layer = myNet.m_layers[layerNum];
		/*weightData<<endl<< "weights for layer " << layerNum << endl;*/
		for (unsigned n = 0; n < layer.size(); ++n){
			
			
			for (size_t w = 2; w < weightArray[rowWeight].size()-1; ++w){
				Neuron &neuron = layer[n];
				neuron.m_outputWeights[w-2].weight =stod( weightArray[rowWeight][w]);
				
			}
			++rowWeight;
		}
	}

};

void Net::saveWeight(Net myNet, string fileName){
	//loop to print all neuron weights but not bias
	ofstream weightData(fileName);
	cout << endl;
	for (unsigned layerNum = 0; layerNum< myNet.m_layers.size()-2; ++layerNum){ //-2 because there are only 2 layers with weight outputs, but not sure why there are 4 layers...
		Layer &layer = myNet.m_layers[layerNum];
		/*weightData<<endl<< "weights for layer " << layerNum << endl;*/
		for (unsigned n = 0; n < layer.size() ; ++n){
			weightData << "Layer: " << layerNum << ", neuron " << n << ", ";
			cout << "Layer: " << layerNum << ", neuron " << n << ", ";
			for (unsigned w = 0; w < layer[n].m_outputWeights.size(); ++w){
				Neuron &neuron = layer[n];
				weightData << neuron.m_outputWeights[w].weight << ", ";
				cout<<neuron.m_outputWeights[w].weight << ", ";
			}
			weightData << endl;
			cout << endl;
		}
	}
	weightData.close();

};

void Net::getResults(vector<double> &resultVals)const{

	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n){
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}

};
Net::Net(const vector<unsigned> topology){
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
		cout << "made a layer: ";
		//made a new layer, now need to add individual neurons
		for (unsigned neuronNum = 0; neuronNum <=topology[layerNum]; neuronNum++){
			m_layers.back().push_back(Neuron(numOutputs, neuronNum)); //putting neuron on most recent layer(back)
			cout << "made a new neuron, ";
		}
		cout << endl;
		// Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
		m_layers.back().back().setOutputVal(1);
	}
}
void Net::feedForward(const vector<double> &inputVals){
	assert(inputVals.size() == m_layers[0].size() - 1); // just checking erros
	for (unsigned i = 0; i < inputVals.size(); ++i){
		m_layers[0][i].setOutputVal(inputVals[i]);
	}
	//forward propagate

	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
		Layer &prevLayer = m_layers[layerNum - 1]; //pointer to previous layer
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n){
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const vector<double> &targetVals){
	//calc overall net error (rms of output errors)
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n){
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta *delta;
	}
	m_error /= outputLayer.size() - 1;// average of error squared
	m_error = sqrt(m_error); //rms
	//implement a recent average measurement


	m_recentAverageError =
		(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
		/ (m_recentAverageSmoothingFactor + 1.0);
	//calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n){
		outputLayer[n].calcOutGradients(targetVals[n]);
	}

	//calculate hiddenl layer gradents
	for (unsigned layerNum = m_layers.size() - 2; layerNum>0; --layerNum){
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];
		for (unsigned n = 0; n < hiddenLayer.size(); ++n){
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}

	}
	//calc for all layers form outputs to first hidden layer update weights
	for (unsigned layerNum = m_layers.size() - 1; layerNum>0; --layerNum){
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < layer.size() - 1; ++n){
			layer[n].updateInputWeights(prevLayer);
		}
	}
}
double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over
