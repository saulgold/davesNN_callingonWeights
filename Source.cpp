#include "TrainingData.h"
#include "Neuron.h"
#include "Net.h"
#include<assert.h>

using namespace std;

void showVectorVals(string label, vector<double> &v){
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		cout << v[i] << " ";
	}

	cout << endl;
}
vector<vector<string> > CSVRead(string CSVIn, string TXTout, int topology[3]){
	
	//first create a vector array of all data
	
	ifstream CSVFile(CSVIn);
	string line,value;
	vector<string> newColumn;
	vector<vector<string> > dataArray;
	unsigned row=0, col=0;
	//get rid of the first line which is text
	getline(CSVFile, line);
	
	//loop through csv
	while (!CSVFile.eof()){

		getline(CSVFile, line);
		istringstream ss(line);
		dataArray.push_back(newColumn);
		col = 0;
		while (!ss.eof()){
			getline(ss, value, ',');
			
			dataArray.at(row).push_back(value);
			cout << dataArray[row][col]<<", ";
			++col;
		}
		cout << endl;
		++row;
	}

	//now export data in the correct format
	ofstream outputFile(TXTout);
	//loop to add topology info
	outputFile << "topology: ";
	for (unsigned i=0; i < 3; ++i){
		cout << topology[i] << " ";
		outputFile << topology[i] << " ";
	}
	cout << endl;
	outputFile << endl;
	//loop to add input and output data  in correct format
	unsigned trainingDataMax = 300;
	for (unsigned i = 0; i < trainingDataMax; ++i){
		outputFile << "in: " << dataArray[i][0]  <<" "<<dataArray[i][1]<< endl << "out: " << dataArray[i][2] << endl;
		//forest fire eg
		//outputFile << "in: " << dataArray[i][8]/*temp*/ << " " << dataArray[i][9]/*RH*/ << " " << dataArray[i][10]/*wind*/ << " " << dataArray[i][11]/*rain*/
		//	<< endl << "out: " << dataArray[i][12]/*area*/<<endl;
	}
	outputFile.close();
	return dataArray;
}


int main(){
	int topologyIn[3] = {2,4,1 };
	vector<vector<string> > dataArray = CSVRead("exor.csv", "initexor.txt", topologyIn);

	TrainingData trainData("init.txt");

	vector<unsigned> topology;
	trainData.getTopology(topology);

	Net myNet(topology);

	vector<double> inputVals, targetVals, resultVals;
	
	int trainingPass = 0;
	ofstream trainingDataSinx("trainingDataexor.txt");
	while (!trainData.isEof()){
		++trainingPass;
		
		cout << endl << "Pass " << trainingPass;
		trainingDataSinx<<endl<< "Pass " << trainingPass;
		//get input data and feed it forward
		if (trainData.getNextInputs(inputVals) != topology[0]){
			break;
		}
		showVectorVals(": inputs:", inputVals);

		trainingDataSinx << endl << ": inputs: ";
		for (unsigned i = 0; i < inputVals.size(); ++i){
			trainingDataSinx << inputVals[i] << " ";
		}
		myNet.feedForward(inputVals);

		//collect the nets actual results
		myNet.getResults(resultVals);
		showVectorVals("outputs:", resultVals);

		trainingDataSinx << endl << ": outputs: ";
		for (unsigned i = 0; i < resultVals.size(); ++i){
			trainingDataSinx << resultVals[i] << " ";
		}
		//train the net what the output should have beed
		trainData.getTargetOutputs(targetVals);
		showVectorVals("targets:", targetVals);
		trainingDataSinx << endl << ": targets: ";
		for (unsigned i = 0; i < targetVals.size(); ++i){
			trainingDataSinx << targetVals[i] << " ";
		}

		


		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		//report how well the training is working, averaged over recent results
		cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;
	}
	trainingDataSinx.close();
	myNet.saveWeight(myNet, "weightValsexor.txt");

	vector<vector<string> > weightArray1;
	myNet.loadWeights(myNet, weightArray1, "weightValsexor.txt");
	
	ofstream resultFile("resultsexor.csv");

	vector<double> predictInput, predictResults;
	resultFile <<"input1,"<<"input2,"<< "predicted Results," << "actual exor" << "% error difference" << endl;
	for (unsigned row = 300; row < 500;++row){
		predictInput.clear();

		predictInput.push_back(stod(dataArray[row][0]));
		predictInput.push_back(stod(dataArray[row][1]));
	
		myNet.feedForward(predictInput);
		myNet.getResults(predictResults);
		resultFile << predictInput[0]<<","<<predictInput[1]<<","<<predictResults[0]<<",";
		resultFile << dataArray[row][2] << "," << 100 * (stod(dataArray[row][2]) - resultVals[0]) / resultVals[0] << endl;
		
//forest file eg
	//	resultFile << predictResults[0] << "," << dataArray[row][12] << "," <<100* (predictResults[0] - stod(dataArray[row][12])) / stod(dataArray[row][12]) << endl;
	}
	
	//loop to test weights

	//while (1){

	//	cout << "enter first input: ";
	//	vector<double> inputVals,resultVals;
	//	double input;
	//	cin>>input;
	//	inputVals.push_back(input);

	//	cout << endl << "enter second input: ";
	//	cin >> input;
	//	inputVals.push_back(input);
	//	myNet.feedForward(inputVals);
	//	myNet.getResults(resultVals);
	//	showVectorVals("outputs: ", resultVals);
	//}
	
	//Neuron::getOutputWeight(
	//cout << endl << "Done";
}