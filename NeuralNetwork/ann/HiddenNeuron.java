package ann;

import java.io.Serializable;

/**
 * For hidden and output layer
 * @author Anand Kumar
 *
 */
class HiddenNeuron extends Neuron implements Serializable 
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public double calculateOutput() {
		double sum = 0;
		for (Edge e : incomingEdges) {
			sum = sum + e.getWeight() * e.getSource().calculateOutput();
		}
		input = sum;
		if (sum == 0)
			input = 1; // for threshold neuron
		output = getSigmoid(input);
		return output;
	}

	@Override
	public double calculateError() {
		return 0;
	}
}