package nn;

public class Axon {
	private Neuron inputNeuron;
	private double weight;
	private double weightChange;
	private double output;
	
	public Axon(Neuron n, double w){
		this.inputNeuron = n;
		this.weight = w;
		this.weightChange = 0;
		this.updateOutput();
	}
	public double getOutput(){
		return this.output;
	}
	public void updateOutput(){
		this.output = this.inputNeuron.getValue() * this.weight;
	}
	public Neuron getInput(){
		return this.inputNeuron;
	}
	public double getWeight(){
		return this.weight;
	}
	public void setWeightChange(double d){
		this.weightChange = d;
		this.weight += this.weightChange;
		this.updateOutput();
	}
}
