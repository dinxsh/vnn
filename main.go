package main

import (
	"fmt"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// Example patterns
	patterns := []Pattern{
		{Features: []float64{0, 0}, MultipleExpectation: []float64{0}},
		{Features: []float64{0, 1}, MultipleExpectation: []float64{1}},
		{Features: []float64{1, 0}, MultipleExpectation: []float64{1}},
		{Features: []float64{1, 1}, MultipleExpectation: []float64{0}},
	}

	// Initialize MLP
	mlp := PrepareMLPNet([]int{2, 2, 1}, 0.5, sigmoid, sigmoidDerivative)

	// Train MLP
	MLPTrain(&mlp, patterns, 10000)

	// Test MLP
	for _, pattern := range patterns {
		outputs := Execute(&mlp, &pattern)
		fmt.Printf("Input: %v, Expected: %v, Output: %v\n", pattern.Features, pattern.MultipleExpectation, outputs)
	}
}
