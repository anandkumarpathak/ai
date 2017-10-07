package ann;

class NeuralNetworkDataSet {
	private double input[][];
	private double output[][];

	public NeuralNetworkDataSet() {

	}

	public void setInput(double input[][]) {
		this.input = input;
	}

	public double[][] getInput() {
		return input;
	}

	public void setOutput(double output[][]) {
		this.output = output;
	}

	public double[][] getOutput() {
		return output;
	}
}