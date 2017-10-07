package ann;

public class Demo {
	
	public static void main(String arg[]) throws Exception {
		// demoLoad();
		demo();
	}

	public static void demo() throws Exception {
		FileParser fp = new FileParser();
		NeuralNetworkDataSet data = fp.parse("D:/AI/characters.txt");
		NeuralNetwork nn = new NeuralNetwork(data.getInput(), data.getOutput());
		// nn.display();
		nn.feedForward();
		try {
			nn.save("D:/AI/ffnn.txt");
			double[] testData = { 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1,
					0, 1, 0, 0, 1, 1, 1, 1, 1, 1 };
			double[] out = nn.test(testData);

			System.out.println("TESTING");
			for (double d : out) {
				System.out.println(d);
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
}