#include "Neuron.h"


double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;


void Neuron::updateInputWeights(Layer &prevLayer){
	//weights to be updated are in the connection struct in the neurons in the preceeding layer
	for (unsigned n = 0; n < prevLayer.size(); ++n){
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
		double newDeltaWeight = //individuatl indput, magnified by the fradient and training rate
			eta * neuron.getOutputVal() *m_gradient
			//also add momentum, which is a fraction of the previous delta
			+ alpha * oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
};
double Neuron::sumDOW(const Layer &nextLayer) const{

	double sum = 0.0;
	//sum our contributions of the rrors at the nodes we feed
	for (unsigned n = 0; n < nextLayer.size() - 1; ++n){
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
};
void Neuron::calcHiddenGradients(const Layer &nextLayer){
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
};
void Neuron::calcOutGradients(double targetVal){
	double delta = targetVal - m_outputVal;
	m_gradient = delta*Neuron::transferFunctionDerivative(m_outputVal);
};
double Neuron::transferFunction(double x){
	//tanh - output range [-1 to 1]
	return tanh(x);
};
double Neuron::transferFunctionDerivative(double x){
	//return tanh derivative
	return 1 - x*x;
};

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
	for (unsigned c = 0; c < numOutputs; ++c){
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}
	m_myIndex = myIndex;
}
void Neuron::feedForward(const Layer &prevLayer){
	double sum = 0.0;
	//sum previous layers outputs
	//and include bias node from previous layer
	for (unsigned n = 0; n < prevLayer.size(); ++n){
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;

	}

	m_outputVal = Neuron::transferFunction(sum);
}
