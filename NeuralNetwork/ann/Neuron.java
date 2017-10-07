package ann;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;

abstract public class Neuron implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public static final double ALPHA = 1.0;
	protected double input; // net input = Summation(WiXi)
	protected double output; // output= f(Summation(WiXi))
	protected double expected;
	protected ArrayList<Edge> incomingEdges = new ArrayList<Edge>();
	
	public static DecimalFormat df = new DecimalFormat("##.#####");

	public Neuron() {
		input = 1;
		output = 0;
		expected = 0;
	}

	public void addEdge(Edge edge) {
		incomingEdges.add(edge);
	}

	public ArrayList<Edge> getIncomingEdges() {
		return incomingEdges;
	}

	abstract public double calculateOutput();

	abstract public double calculateError();

	public void setExpected(double expected) {
		this.expected = expected;
	}

	public void setInput(double input) {

		this.input = input;
	}

	public void setOutput(double output) {
		this.output = output;
	}

	public double getExpected() {
		return expected;
	}

	public double getInput() {
		return input;
	}

	public double getOutput() {
		return output;
	}

	protected double f(double input) {
		return 0;
	}

	private double truncate(double sum) {
		return Double.parseDouble(df.format(sum));
		// return sum;
	}

	protected double getSigmoid(double sum) {
		return truncate(1 / (1 + Math.exp(-1 * ALPHA * sum)));
	}

	public double getGradient() {
		return output * (1 - output);
	}

}