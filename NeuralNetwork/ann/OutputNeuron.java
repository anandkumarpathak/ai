package ann;

import java.io.Serializable;

class OutputNeuron extends HiddenNeuron implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public double calculateError() {
		return expected - output;
	}

	public double calculateOutput() {
		double sum = 0;
		for (Edge e : incomingEdges) {
			sum = sum + e.getWeight() * e.getSource().getOutput();
		}
		input = sum;
		if (sum == 0)
			input = -1; // for threshold neuron
		output = getSigmoid(input);
		return output;
	}

}