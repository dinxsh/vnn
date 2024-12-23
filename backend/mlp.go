package main

import (
	"fmt"
	"math"
	"math/rand"
)

type Pattern struct {
	Features            []float64 `json:"features"`
	MultipleExpectation []float64 `json:"multipleExpectation"`
}

type NeuronUnit struct {
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
	Lrate   float64   `json:"lrate"`
	Value   float64   `json:"value"`
	Delta   float64   `json:"delta"`
}

type NeuralLayer struct {
	NeuronUnits []NeuronUnit `json:"neuronUnits"`
	Length      int          `json:"length"`
}

type MultiLayerNetwork struct {
	L_rate       float64       `json:"lRate"`
	NeuralLayers []NeuralLayer `json:"neuralLayers"`
	T_func       func(float64) float64
	T_func_d     func(float64) float64
}

func RandomNeuronInit(n *NeuronUnit, numInputs int) {
	n.Weights = make([]float64, numInputs)
	for i := range n.Weights {
		n.Weights[i] = rand.Float64()*2 - 1 // Initialize between -1 and 1
	}
	n.Bias = rand.Float64()*2 - 1
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

	// Set input layer values
	for i, value := range pattern.Features {
		mlp.NeuralLayers[0].NeuronUnits[i].Value = value
	}

	// Forward propagation
	for i := 1; i < len(mlp.NeuralLayers); i++ {
		layer := &mlp.NeuralLayers[i]
		prevLayer := &mlp.NeuralLayers[i-1]
		
		for j := range layer.NeuronUnits {
			neuron := &layer.NeuronUnits[j]
			sum := neuron.Bias
			
			for k, weight := range neuron.Weights {
				sum += weight * prevLayer.NeuronUnits[k].Value
			}
			
			neuron.Value = mlp.T_func(sum)
		}
	}

	// Get output values
	outputLayer := mlp.NeuralLayers[len(mlp.NeuralLayers)-1]
	outputs := make([]float64, len(outputLayer.NeuronUnits))
	for i, neuron := range outputLayer.NeuronUnits {
		outputs[i] = neuron.Value
	}
	
	return outputs, nil
}

func BackPropagate(mlp *MultiLayerNetwork, pattern *Pattern, outputs []float64) float64 {
	totalError := 0.0
	
	// Calculate output layer deltas and error
	outputLayer := &mlp.NeuralLayers[len(mlp.NeuralLayers)-1]
	for i := range outputLayer.NeuronUnits {
		error := pattern.MultipleExpectation[i] - outputs[i]
		totalError += error * error
		neuron := &outputLayer.NeuronUnits[i]
		neuron.Delta = error * mlp.T_func_d(neuron.Value)
	}

	// Calculate hidden layer deltas
	for l := len(mlp.NeuralLayers) - 2; l >= 0; l-- {
		layer := &mlp.NeuralLayers[l]
		nextLayer := &mlp.NeuralLayers[l+1]
		
		for i := range layer.NeuronUnits {
			error := 0.0
			for j := range nextLayer.NeuronUnits {
				error += nextLayer.NeuronUnits[j].Delta * nextLayer.NeuronUnits[j].Weights[i]
			}
			neuron := &layer.NeuronUnits[i]
			neuron.Delta = error * mlp.T_func_d(neuron.Value)
		}
	}

	// Update weights and biases
	for l := 1; l < len(mlp.NeuralLayers); l++ {
		layer := &mlp.NeuralLayers[l]
		prevLayer := &mlp.NeuralLayers[l-1]
		
		for i := range layer.NeuronUnits {
			neuron := &layer.NeuronUnits[i]
			
			// Update weights
			for j := range neuron.Weights {
				neuron.Weights[j] += mlp.L_rate * neuron.Delta * prevLayer.NeuronUnits[j].Value
			}
			
			// Update bias
			neuron.Bias += mlp.L_rate * neuron.Delta
		}
	}

	return totalError / float64(len(outputs))
}

func MLPTrain(mlp *MultiLayerNetwork, patterns []Pattern, epochs int) float64 {
	var finalError float64
	for i := 0; i < epochs; i++ {
		totalError := 0.0
		for _, pattern := range patterns {
			outputs, err := Execute(mlp, &pattern)
			if err != nil {
				continue
			}
			error := BackPropagate(mlp, &pattern, outputs)
			totalError += error
		}
		finalError = totalError / float64(len(patterns))
	}
	return finalError
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}
