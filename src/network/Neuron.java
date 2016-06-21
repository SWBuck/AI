package network;

import java.util.ArrayList;

public class Neuron {
	private double net;
	private double value;
	private ArrayList<Axon> inputs;
	private double biasWeight;
	private double biasWeightChange;
	private double errorSignal;
	
	//Input Neuron//
	public Neuron(double v){
		this.value = v;
	}
	//Other Neurons//
	public Neuron(){
		this.inputs = new ArrayList<Axon>();
	}
	public void setBias(double b){
		this.biasWeight = b;
	}
	public void setValue(double v){
		this.value = v;
	}
	public double getValue(){
		return this.value;
	}
	public void addAxon(Axon a){
		this.inputs.add(a);
	}
	public void update(){
		this.updateBias();
		this.updateAxons();
		this.updateNet();
		this.updateValue();
	}

	private void updateBias(){
		this.biasWeight += this.biasWeightChange;
	}
	private void updateAxons(){
		for(Axon a : this.inputs){
			a.updateOutput();
		}
	}
	private void updateNet(){
		this.net = 0;
		this.net += this.biasWeight;
		for(Axon a : this.inputs){
			this.net += a.getOutput();
		}
	}
	private void updateValue(){
		this.value = sigmoid(this.net);
	}
	private double sigmoid(double x){
		return 1/(1+Math.pow(Math.E, (x*-1)));
	}
	public void setBiasWeightChange(double d){
		this.biasWeightChange = d;
	}
	public void setErrorSignal(double d){
		this.errorSignal = d;
	}
	public double getErrorSignal(){
		return this.errorSignal;
	}
	public ArrayList<Axon> getInputs(){
		return this.inputs;
	}
	public double getBiasWeight(){
		return this.biasWeight;
	}
}
