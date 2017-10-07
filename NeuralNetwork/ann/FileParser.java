package ann;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.StringTokenizer;

class FileParser {

	protected String[] getTokens(String line, String delimiter) {
		String array[];
		StringTokenizer tokenizer = new StringTokenizer(line, delimiter);
		array = new String[tokenizer.countTokens()];
		int i = 0;
		while (tokenizer.hasMoreTokens()) {
			array[i++] = tokenizer.nextToken();
		}
		return array;
	}

	protected double[] toDouble(String[] array) {
		double d[] = new double[array.length];
		int i = 0;
		for (String s : array) {
			d[i++] = Double.valueOf(s);
		}
		return d;
	}

	protected double[][] listTo2DArray(ArrayList<double[]> list)
			throws Exception {
		double[][] inputValues;
		inputValues = new double[list.size()][];
		int row = 0;
		for (double[] inp : list) {
			inputValues[row] = inp;
			if (row > 0
					&& inputValues[row - 1].length != inputValues[row].length)
				throw new Exception("Invalid dataset, Input sizes vary !");
			row++;
		}
		return inputValues;
	}

	public void showArray(String[] arr) {
		System.out.println();
		for (String s : arr) {
			System.out.print(s);
		}
	}

	public NeuralNetworkDataSet parse(String trainingDataFile) throws Exception {
		NeuralNetworkDataSet dataSet = null;
		BufferedReader br = new BufferedReader(new FileReader(trainingDataFile));
		String line = null;
		int row = 0;
		double[] input = null, output = null;
		ArrayList<double[]> inputArray, outputArray;
		inputArray = new ArrayList<double[]>();
		outputArray = new ArrayList<double[]>();

		while ((line = br.readLine()) != null) {

			String data[] = getTokens(line, ",");
			if ((row % 2) == 0) {
				input = toDouble(data);
				output = null;
			} else {
				output = toDouble(data);
				inputArray.add(input);
				outputArray.add(output);
				input = null;
			}
			showArray(data);
			row++;
		}
		if (output == null)
			throw new Exception("Invalid dataset\nNo Data or insufficient data");
		double[][] inputValues, outputValues;
		inputValues = listTo2DArray(inputArray);
		outputValues = listTo2DArray(outputArray);
		dataSet = new NeuralNetworkDataSet();
		dataSet.setInput(inputValues);
		dataSet.setOutput(outputValues);
		return dataSet;
	}
}