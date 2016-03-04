#pragma once
#include "Net.h"
#include<vector>

class Neuron;
typedef vector<Neuron> Layer;
struct Connection{
	double weight;
	double deltaWeight;
};

class Neuron
{


public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val){ m_outputVal = val; }
	double getOutputVal(void) const{ return m_outputVal; }

	void feedForward(const Layer &prevLayer);
	void calcOutGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	double getOutputWeight(void);

	vector<Connection> m_outputWeights; // used to be private but trying public to if makes difference

private:
	static double eta; // [0 to 1] overall net training rate
	static double alpha; //[0 to n] mulitplier of last weight change
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void){ return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;

	unsigned m_myIndex;
	double m_gradient;
};

