package ann;

import java.io.Serializable;
import java.util.ArrayList;

public class Layer implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	ArrayList<Neuron> neurons = new ArrayList<Neuron>();

	public void addNeuron(Neuron neuron) {
		neurons.add(neuron);
	}

	public ArrayList<Neuron> getNeurons() {
		return neurons;
	}

	public void display() {
		String details = "";
		for (Neuron neuron : neurons) {
			details = details + "(" + neuron.getOutput() + ")" + "\t"; // neuron.getClass().getName()
		}
		System.out.println(details);
	}
}
