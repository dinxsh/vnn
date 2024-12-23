package main

import (
	"fmt"
	"math"
	"math/rand"
)

type Pattern struct {
	Features            []float64
	MultipleExpectation []float64
}

type NeuronUnit struct {
	Weights []float64
	Bias    float64
	Lrate   float64
	Value   float64
	Delta   float64
}

type NeuralLayer struct {
	NeuronUnits []NeuronUnit
	Length      int
}

type MultiLayerNetwork struct {
	L_rate       float64
	NeuralLayers []NeuralLayer
	T_func       func(float64) float64
	T_func_d     func(float64) float64
}

func RandomNeuronInit(n *NeuronUnit, numInputs int) {
	n.Weights = make([]float64, numInputs)
	for i := range n.Weights {
		n.Weights[i] = rand.Float64()
	}
	n.Bias = rand.Float64()
	n.Lrate = 0.1
}

func PrepareLayer(numNeurons, numInputs int) NeuralLayer {
	layer := NeuralLayer{NeuronUnits: make([]NeuronUnit, numNeurons), Length: numNeurons}
	for i := 0; i < numNeurons; i++ {
		RandomNeuronInit(&layer.NeuronUnits[i], numInputs)
	}
	return layer
}

func PrepareMLPNet(layerSizes []int, learningRate float64, transferFunc, transferFuncDeriv func(float64) float64) MultiLayerNetwork {
	mlp := MultiLayerNetwork{
		L_rate:       learningRate,
		T_func:       transferFunc,
		T_func_d:     transferFuncDeriv,
		NeuralLayers: make([]NeuralLayer, len(layerSizes)),
	}
	for i, size := range layerSizes {
		if i == 0 {
			mlp.NeuralLayers[i] = PrepareLayer(size, 0)
		} else {
			mlp.NeuralLayers[i] = PrepareLayer(size, layerSizes[i-1])
		}
	}
	return mlp
}

func Execute(mlp *MultiLayerNetwork, pattern *Pattern) ([]float64, error) {
	if len(pattern.Features) != len(mlp.NeuralLayers[0].NeuronUnits) {
		return nil, fmt.Errorf("input feature size does not match the input layer size")
	}

	inputs := pattern.Features
	for _, layer := range mlp.NeuralLayers {
		outputs := make([]float64, layer.Length)
		for j, neuron := range layer.NeuronUnits {
			sum := neuron.Bias
			for k, weight := range neuron.Weights {
				sum += weight * inputs[k]
			}
			outputs[j] = mlp.T_func(sum)
			layer.NeuronUnits[j].Value = outputs[j]
		}
		inputs = outputs
	}
	return inputs, nil
}

func BackPropagate(mlp *MultiLayerNetwork, pattern *Pattern, outputs []float64) {
	outputLayer := &mlp.NeuralLayers[len(mlp.NeuralLayers)-1]
	for i, neuron := range outputLayer.NeuronUnits {
		error := pattern.MultipleExpectation[i] - outputs[i]
		neuron.Delta = error * mlp.T_func_d(neuron.Value)
	}

	for l := len(mlp.NeuralLayers) - 2; l >= 0; l-- {
		layer := &mlp.NeuralLayers[l]
		nextLayer := &mlp.NeuralLayers[l+1]
		for i, neuron := range layer.NeuronUnits {
			error := 0.0
			for _, nextNeuron := range nextLayer.NeuronUnits {
				error += nextNeuron.Weights[i] * nextNeuron.Delta
			}
			neuron.Delta = error * mlp.T_func_d(neuron.Value)
		}
	}

	for l := 1; l < len(mlp.NeuralLayers); l++ {
		inputs := mlp.NeuralLayers[l-1].NeuronUnits
		layer := &mlp.NeuralLayers[l]
		for _, neuron := range layer.NeuronUnits {
			for j, inputNeuron := range inputs {
				neuron.Weights[j] += mlp.L_rate * neuron.Delta * inputNeuron.Value
			}
			neuron.Bias += mlp.L_rate * neuron.Delta
		}
	}
}

func MLPTrain(mlp *MultiLayerNetwork, patterns []Pattern, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		for _, pattern := range patterns {
			outputs, _ := Execute(mlp, &pattern)
			BackPropagate(mlp, &pattern, outputs)
		}
	}
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}
