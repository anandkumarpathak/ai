package ann;

import java.io.Serializable;

public class Edge implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private Neuron source;
	private double weight;
	private double delta;

	public Edge(double weight) {
		this.weight = weight;
		this.delta = 0;
	}

	public Neuron getSource() {
		return source;
	}

	public void setSource(Neuron source) {
		this.source = source;
	}

	public double getDelta() {
		return delta;
	}

	public void setDelta(double delta) {
		this.delta = delta;
	}

	public double getWeight() {
		return weight;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}

	public String toString() {
		return "" + weight;
	}

}
