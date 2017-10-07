package ann;

import java.io.Serializable;

/**
 * For input layer
 * @author Anand Kumar
 *
 */
class InputNeuron extends Neuron implements Serializable //
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public InputNeuron(double input) {
		this.input = input;
		this.output = input;
	}

	public void setInput(double input) {
		this.output = input;
		this.input = input;
	}

	public double calculateOutput() {

		return input;
	}

	@Override
	public double calculateError() {
		return 0;
	}

}