package network;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class Network {
	private ArrayList<Neuron> inputLayer;
	private ArrayList<Neuron> hiddenLayer;
	private ArrayList<Neuron> outputLayer;
	private double[][] trainingInputSet;

	private double[][] trainingOutputSet;
	private double targetRMSE = 0.0001;
	private double learningRate = 0.75;
	private ArrayList<Integer> firingOrder;

	public Network(int inputSize, int hiddenSize, double[][] trainIn,
			double[][] trainOut) {
		System.out.println("Epochs\tRMSE");
		this.trainingInputSet = trainIn;
		this.trainingOutputSet = trainOut;
		this.firingOrder = new ArrayList<Integer>();
		for (int i = 0; i < this.trainingOutputSet.length; i++) {
			this.firingOrder.add(i);
		}
		this.inputLayer = new ArrayList<Neuron>();
		this.hiddenLayer = new ArrayList<Neuron>();
		this.outputLayer = new ArrayList<Neuron>();

		for (int i = 0; i < inputSize; i++) {
			inputLayer.add(new Neuron());
		}
		for (int i = 0; i < hiddenSize; i++) {
			Neuron hidden = new Neuron();
			hidden.setBias(-1 + (this.randomDouble(-1.0, 1.0)));
			for (Neuron a : this.inputLayer) {
				double randWeight = -1 + (this.randomDouble(-1.0, 1.0));
				hidden.addAxon(new Axon(a, randWeight));
			}
			this.hiddenLayer.add(hidden);
		}
		// 1 output neuron
		Neuron o = new Neuron();
		o.setBias(-1 + (this.randomDouble(-1.0, 1.0)));
		for (Neuron a : this.hiddenLayer) {
			double randWeight = -1 + (this.randomDouble(-1.0, 1.0));
			o.addAxon(new Axon(a, randWeight));
		}
		this.outputLayer.add(o);
	}

	private double randomDouble(double min, double max) {
		return min + (max - min) * new Random().nextDouble();
	}

	public void print() {
		System.out.println("Output");
		for (Neuron a : this.outputLayer) {
			System.out.println(a.getValue());
		}
	}

	public double[] getOutput() {
		double[] out = new double[this.outputLayer.size()];
		for (int i = 0; i < out.length; i++) {
			out[i] = this.outputLayer.get(i).getValue();
		}
		return out;
	}

	public void feedForward(double[] input) {
		for (int i = 0; i < input.length; i++) {
			this.inputLayer.get(i).setValue(input[i]);
		}
		for (Neuron a : this.hiddenLayer) {
			a.update();
		}
		for (Neuron a : this.outputLayer) {
			a.update();
		}
	}

	public void backProp(double[] desiredOut) {
		for (Neuron out : this.outputLayer) {
			out.setErrorSignal((desiredOut[this.outputLayer.indexOf(out)] - out
					.getValue()) * out.getValue() * (1 - out.getValue()));
			out.setBiasWeightChange(this.learningRate * out.getErrorSignal());
			for (Axon a : out.getInputs()) {
				a.getInput().setErrorSignal(
						a.getInput().getValue() * (1 - a.getInput().getValue())
								* out.getErrorSignal() * a.getWeight());
				a.setWeightChange(this.learningRate * out.getErrorSignal()
						* a.getInput().getValue());
			}
		}
		for (Neuron a : this.hiddenLayer) {
			a.setBiasWeightChange(this.learningRate * a.getErrorSignal());
			for (Axon ax : a.getInputs()) {
				ax.setWeightChange(this.learningRate * a.getErrorSignal()
						* ax.getInput().getValue());
			}
		}
	}

	public void train() {
		//Try using Accuracy rather than RMSE
		double currentRMSE = 1.0;
		int count = 0;
		while (currentRMSE > this.targetRMSE) {
			Collections.shuffle(this.firingOrder);
			for (Integer in : firingOrder) {
				
				this.feedForward(this.trainingInputSet[in]);
				this.backProp(this.trainingOutputSet[in]);
			}
			count++;
			if (count % 1000 == 0) {
				currentRMSE = this.calcRMSE();
				System.out.println(count+"\t"+currentRMSE);
			}
			if(count % 10000 == 0){
				

			}
		}
		System.out.println("RMSE: " + currentRMSE);
		System.out.println("Epochs: " + count);
	}

	private double calcRMSE() {
		double currentTSSE = this.calcTSSE();
		return Math
				.sqrt((2 * currentTSSE)
						/ (this.trainingInputSet.length * this.trainingOutputSet[0].length));
	}

	private double calcTSSE() {
		double sum = 0;
		for (Integer i : this.firingOrder) {
			this.feedForward(this.trainingInputSet[i]);
			for (int j = 0; j < this.trainingOutputSet[i].length; j++) {
				sum += Math
						.pow((this.trainingOutputSet[i][j] - this.getOutput()[j]),
								2);
			}
		}
		return .5 * sum;
	}

	public void test() {
		double[][] result = new double[22][22];
		int iCount = 0;
		for (double i = 0.0; i < 1.0; i += 0.05) {
			int jCount = 0;
			for (double j = 0.0; j < 1.0; j += 0.05) {
				this.feedForward(new double[] { i, j });
				result[iCount][jCount] = this.getOutput()[0];
				jCount++;
			}
			iCount++;
		}
		for (int x = 0; x < result.length; x++) {
			System.out.println(x + "\t");
			for (int y = 0; y < result.length; y++) {
				System.out.print(result[x][y] + "\t");
			}
		}
	}

	public void setTrainingInputSet(double[][] trainingInputSet) {
		this.trainingInputSet = trainingInputSet;
	}

	public void setTrainingOutputSet(double[][] trainingOutputSet) {
		this.trainingOutputSet = trainingOutputSet;
	}

	public static void main(String[] args) {
		// XOR Example
		double[][] i = new double[][] { { 0.1, 0.1 }, { 0.1, 0.9 },
				{ 0.9, 0.1 }, { 0.9, 0.9 }, };
		double[][] o = new double[][] { { 0.1 }, { 0.9 }, { 0.9 }, { 0.1 } };
		Network nn = new Network(2, 2, i, o);
		nn.train();

		nn.feedForward(new double[] { 0.89, 0.11 });
		System.out.println("0.89\t0.11\t" + nn.getOutput()[0]);

		nn.feedForward(new double[] { 0.1, 0.1 });
		System.out.println("0.1\t0.1\t" + nn.getOutput()[0]);

		nn.feedForward(new double[] { 0.9, 0.9 });
		System.out.println("0.9\t0.9\t" + nn.getOutput()[0]);

		nn.feedForward(new double[] { 0.1, 0.9 });
		System.out.println("0.1\t0.9\t" + nn.getOutput()[0]);

		nn.feedForward(new double[] { 0.0001, 0.0001 });
		System.out.println("0.0001\t0.0001\t" + nn.getOutput()[0]);
		System.out.println();
		System.out.println();
	}
}
