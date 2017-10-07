package ann;

import java.util.HashMap;

class NeuralNetworkDataSetList {
	private HashMap<double[], double[]> dataSet;

	public NeuralNetworkDataSetList() {
		// TODO Auto-generated constructor stub
	}

	public void setDataSet(HashMap<double[], double[]> dataSet) {
		this.dataSet = dataSet;
	}

	public HashMap<double[], double[]> getDataSet() {
		return dataSet;
	}
}