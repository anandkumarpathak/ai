/**
 * Java Library for Neural Network
 * @author Anand Kumar
 * 
 */
package ann;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;

public class NeuralNetwork implements Serializable {
	private static final long serialVersionUID = -3554215898230072544L;
	Layer input = new Layer();
	Layer hidden = new Layer();
	Layer output = new Layer();
	ArrayList<Edge> edges = new ArrayList<Edge>();

	double inputValues[][];
	double outputValues[][];

	DecimalFormat df = new DecimalFormat("##.#####");

	public static double RATE = 0.75;// 0.8; // Learning rate

	Random ran = new Random();

	private int epochCount = 1;

	public static double ALPHA = 0.5;// 0.25; // Momentum

	public boolean validateData() {
		if (inputValues.length != outputValues.length)
			return false;
		int count = inputValues[0].length;
		for (double[] data : inputValues) {
			if (data.length != count)
				return false;
		}
		count = outputValues[0].length;
		for (double[] data : outputValues) {
			if (data.length != count)
				return false;
		}
		return true;
	}

	public NeuralNetwork(double input[][], double expected[][])
			throws Exception {
		this.inputValues = input;
		this.outputValues = expected;
		if (!validateData())
			throw new Exception("Invalid Data set");
		initialize();
	}

	private void initialize() {
		initializeInputLayer();
		initializeHiddenLayer();
		initializeOutputLayer();
		initializeInputHiddenWeights();
		initializeHiddenOutputWeights();
	}

	private void initializeInputLayer() {
		for (int i = 0; i < inputValues[0].length; i++) {
			Neuron neuron = new InputNeuron(0);
			input.addNeuron(neuron);
		}
		Neuron neuron = new InputNeuron(1); // For threshold
		input.addNeuron(neuron);
	}

	private void initializeHiddenLayer() {
		for (int i = 0; i < inputValues[0].length; i++) {
			Neuron neuron;
			neuron = new HiddenNeuron();
			hidden.addNeuron(neuron);
		}
		Neuron neuron = new InputNeuron(1);
		hidden.addNeuron(neuron);
	}

	// ////////////// here
	private void initializeOutputLayer() {
		for (int i = 0; i < outputValues[0].length; i++) {
			Neuron neuron;
			neuron = new OutputNeuron();
			output.addNeuron(neuron);
		}
	}

	Random random = new Random();

	public double truncate(double output) {
		return Double.parseDouble(df.format(output));
		// return output;
	}

	private double getRandom() {
		random.setSeed(System.nanoTime());
		return truncate(random.nextDouble());
	}

	private Edge getEdge() {
		ran.setSeed(System.nanoTime());
		int i = ran.nextInt();
		return new Edge(0.25 * Math.abs(i) / i);
	}

	private void initializeInputHiddenWeights() {
		for (int j = 0; j < hidden.getNeurons().size() - 1; j++) // for each
																	// hidden
																	// neuron -
																	// 1
		{
			for (int i = 0; i < input.getNeurons().size(); i++) // for each
																// input neuron
			{
				Edge edge = getEdge();
				edge.setSource(input.getNeurons().get(i));
				hidden.getNeurons().get(j).addEdge(edge);
				edges.add(edge);
			}
		}
	}

	private void initializeHiddenOutputWeights() {
		for (int k = 0; k < output.getNeurons().size(); k++) // for each hidden
																// neuron - 1
		{
			for (int j = 0; j < hidden.getNeurons().size(); j++) // for each
																	// input
																	// neuron
			{
				Edge edge = getEdge();
				edge.setSource(hidden.getNeurons().get(j));
				output.getNeurons().get(k).addEdge(edge);
				edges.add(edge);
			}
		}
	}

	public void run() {
		double oldDelta, newDelta, weight;
		double totalDelta;
		for (Neuron neuron : hidden.getNeurons())
			neuron.calculateOutput();
		for (Neuron neuron : output.getNeurons())
			neuron.calculateOutput();
		for (int k = 0; k < output.getNeurons().size(); k++) {
			for (int j = 0; j < output.getNeurons().get(k).getIncomingEdges()
					.size(); j++) {
				Edge edge = output.getNeurons().get(k).getIncomingEdges()
						.get(j);
				weight = edge.getWeight();
				oldDelta = edge.getDelta();
				newDelta = calculateWeightDeltaJK(j, k);
				edge.setDelta(newDelta); // set new delta
				totalDelta = newDelta + ALPHA * oldDelta;
				weight = weight + totalDelta;
				edge.setWeight(truncate(weight));
			}
		}
		//
		for (int j = 0; j < hidden.getNeurons().size(); j++) {
			for (int i = 0; i < hidden.getNeurons().get(j).getIncomingEdges()
					.size(); i++) {
				Edge edge = hidden.getNeurons().get(j).getIncomingEdges()
						.get(i);
				weight = edge.getWeight();
				oldDelta = edge.getDelta();
				newDelta = calculateWeightDeltaIJ(i, j);
				edge.setDelta(newDelta); // set new delta
				totalDelta = newDelta + ALPHA * oldDelta;
				// System.out.println("NewDelta "+newDelta+ " TotalDelta " +
				// totalDelta);
				weight = weight + totalDelta;
				edge.setWeight(truncate(weight));
			}
		}
	}

	public double calculateTotalError() {
		double total = 0;
		for (Neuron neuron : output.getNeurons()) {
			total = total + neuron.calculateError() * neuron.calculateError();
		}
		return Math.sqrt(total) / output.getNeurons().size(); // Root Mean
																// Square
	}

	public boolean convergedAll() {
		double error = 0;
		for (int n = 0; n < inputValues.length; n++) {
			initializeInputOutputNeurons(n);
			for (Neuron neuron : hidden.getNeurons())
				neuron.calculateOutput();
			for (Neuron neuron : output.getNeurons())
				neuron.calculateOutput();
			error = error + calculateTotalError();
			// System.out.println("Error at " + n +" is "+
			// error+" Expected "+output.getNeurons().get(0).getExpected()+" Actual "+output.getNeurons().get(0).getOutput()
			// );
		}
		if (error > 0.1)
			return false;
		return true;
	}

	public void epoch() {
		double error;
		boolean converged = false;
		int count = 0;
		int epochCount = 0;
		while (!converged) {
			System.out.println("Epoch " + ++epochCount);
			for (int n = 0; n < inputValues.length; n++) {
				initializeInputOutputNeurons(n);
				while (true) {
					run();
					error = calculateTotalError();
					count++;
					if (error <= 0.1) {
						// System.out.println("Converged for "+n+" Output "+output.getNeurons().get(0).getOutput()+" Iteration "+count);
						break;
					}

				}

				count = 0;
			}
			// display();
			converged = convergedAll();
			if ((++epochCount % 100) == 0) {
				RATE = RATE + 0.025;
				ALPHA = ALPHA + 0.025;
			}
		}
		System.out.println("Converged for All");

	}

	private void initializeInputOutputNeurons(int n) // input number
	{
		int i;
		int k;
		for (i = 0; i < input.getNeurons().size() - 1; i++) // for each input
															// neuron
		{
			Neuron neuron = input.getNeurons().get(i);
			neuron.setInput(inputValues[n][i]);
		}
		input.getNeurons().get(i).setInput(1); // for threshhold
		hidden.getNeurons().get(hidden.getNeurons().size() - 1).setInput(1); // for
																				// threshhold
		for (k = 0; k < output.getNeurons().size(); k++) // for each output
															// neuron
		{
			Neuron neuron = output.getNeurons().get(k);
			neuron.setExpected(outputValues[n][k]);
		}
	}

	void feedForward() {
		epoch();
	}

	public void display() // Current weights
	{
		/*
		 * System.out.println("Input"); input.display();
		 * System.out.println("Hidden "); hidden.display();
		 * System.out.println("Output "); output.display();
		 */
		System.out.println("Weights");
		for (int j = 0; j < hidden.getNeurons().size() - 1; j++) // for each
																	// input
																	// neuron
		{
			Neuron neuron = hidden.getNeurons().get(j);
			for (Edge edge : neuron.getIncomingEdges()) {
				System.out.print(edge + "\t");
			}
			System.out.println();
		}
		for (int k = 0; k < output.getNeurons().size(); k++) // for each hidden
																// neuron - 1
		{
			for (Edge edge : output.getNeurons().get(k).getIncomingEdges()) {
				System.out.print(edge + "\t");
			}
			System.out.println();
		}
		/*
		 * System.out.println("Hidden "); hidden.display();
		 */
		// System.out.println("Output ");
		// output.display();

	}

	// Weight(i,j) = Weight(i,j) + R * Delta * Yj

	public double calculateDeltaK(int k) // for K'th output neuron
	{
		Neuron neuron = output.getNeurons().get(k);
		return neuron.getGradient() * neuron.calculateError();
	}

	public double calculateDeltaJ(int j) {
		double summationK = 0;
		Neuron neuron = hidden.getNeurons().get(j);
		for (int k = 0; k < output.getNeurons().size(); k++) {
			// System.out.println("DeltaJ");
			summationK = summationK + getWeightJK(j, k) * calculateDeltaK(k);
			// System.out.println("DeltaJ "+summationK);
		}
		// System.out.println(" Grad "+ neuron.getGradient());
		return neuron.getGradient() * summationK; // Yj*(1-Yj)*Summation(W[j][k]*Delta[K])
	}

	public double calculateWeightDeltaJK(int j, int k) {
		return RATE * hidden.getNeurons().get(j).getOutput()
				* calculateDeltaK(k);
	}

	public double calculateWeightDeltaIJ(int i, int j) {
		// System.out.println("i "+input.getNeurons().get(i).getOutput());
		return RATE * input.getNeurons().get(i).getOutput()
				* calculateDeltaJ(j);
	}

	public double getWeightJK(int j, int k) {
		for (int kk = 0; kk < output.getNeurons().size(); kk++) {
			if (kk == k) {
				for (int jj = 0; jj < output.getNeurons().get(k)
						.getIncomingEdges().size(); jj++) {
					if (jj == j) {
						Neuron neuron = output.getNeurons().get(k);
						return neuron.getIncomingEdges().get(j).getWeight();
					}
				}
			}
		}
		// System.out.println("Failed...");
		return 0;
	}

	public double getWeightIJ(int i, int j) {
		for (int jj = 0; jj < hidden.getNeurons().size(); jj++) {
			if (jj == j) {
				for (int ii = 0; ii < hidden.getNeurons().get(j)
						.getIncomingEdges().size(); ii++) {
					if (ii == i) {
						Neuron neuron = hidden.getNeurons().get(j);
						return neuron.getIncomingEdges().get(i).getWeight();
					}
				}
			}
		}
		// System.out.println("Failed...ij");
		return 0;
	}

	public void save(String file) throws Exception {
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(
				file));
		oos.writeObject(this);
		oos.flush();
		oos.close();
	}

	public static NeuralNetwork load(String file) throws Exception {
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
		NeuralNetwork nn = (NeuralNetwork) ois.readObject();
		ois.close();
		return nn;
	}

	public double[] test(double[] test) {
		double[] out = new double[output.getNeurons().size()];
		int i = 0;
		for (i = 0; i < input.getNeurons().size() - 1; i++) {
			input.getNeurons().get(i).setInput(test[i]);
		}
		for (Neuron neuron : hidden.getNeurons())
			neuron.calculateOutput();
		i = 0;
		for (Neuron neuron : output.getNeurons()) {
			neuron.calculateOutput();
			out[i++] = neuron.getOutput();
		}
		return out;
	}

	/**
	 * The file contains the training data in format comma seperated input :
	 * comma seperated output 1,0.1,1.1,2.5 1,0 1,1.1,0.1,2.0 0,0
	 * 
	 * @param trainingDataFile
	 * @throws Exception
	 */
	public NeuralNetwork(String trainingDataFile) throws Exception {

	}

	public static void demo1() throws Exception {
		double inputValues[][] = { // x ^ y
		{ 0.001, 0 }, // -1 is added by the network, so it becomes 0,0,-1
				{ 0, 1.22 }, // same way here
				{ 1, 0 }, { 1, 1 } };
		double expectedOutput[][] = { //
		{ 0 }, { 1 }, { 1 }, { 1 } };

		NeuralNetwork nn = new NeuralNetwork(inputValues, expectedOutput);
		// nn.display();
		nn.feedForward();
		double[] testData = { 1, 0.01 };
		double[] out = nn.test(testData);

		System.out.println("TESTING");
		for (double d : out) {
			System.out.println(d);
		}
	}

	public double variance(double output[], double expected[]) {
		double var = 0;
		for (int i = 0; i < output.length; i++) {
			if (expected[i] == 0) {
				var = var + (0.1 - output[i]);
			} else {
				var = var + (output[i] - 0.9);
			}

		}
		return var;
	}

	public static void demoLoad() throws Exception {
		NeuralNetwork nn = NeuralNetwork.load("D:/AI/ffnn.txt");
		try {
			double[] testData = { 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
					0, 1, 0, 0, 1, 1, 1, 1, 1, 1 };
			double[] out = nn.test(testData);
			double[] expected = { 0, 0, 1, 0 };
			System.out.println("TESTING");
			for (double d : out) {
				System.out.println(d);
			}
			System.out.println(nn.variance(out, expected));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	

}