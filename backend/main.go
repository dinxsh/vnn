package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"sync"
	"time"
	"github.com/gorilla/mux"
	"github.com/rs/cors"
)

type NetworkState struct {
	Layers []LayerState `json:"layers"`
	Error  float64      `json:"error"`
	Epoch  int         `json:"epoch"`
}

type LayerState struct {
	Neurons []NeuronState `json:"neurons"`
}

type NeuronState struct {
	Weights []float64 `json:"weights"`
	Value   float64   `json:"value"`
	Bias    float64   `json:"bias"`
}

type TrainingRequest struct {
	Patterns []Pattern `json:"patterns"`
	Epochs   int      `json:"epochs"`
}

var (
	mlp           MultiLayerNetwork
	currentState  NetworkState
	trainingError float64
	currentEpoch  int
	mu            sync.RWMutex
)

func getNetworkState(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	defer mu.RUnlock()

	state := NetworkState{
		Error: trainingError,
		Epoch: currentEpoch,
	}
	
	for _, layer := range mlp.NeuralLayers {
		layerState := LayerState{}
		for _, neuron := range layer.NeuronUnits {
			layerState.Neurons = append(layerState.Neurons, NeuronState{
				Weights: neuron.Weights,
				Value:   neuron.Value,
				Bias:    neuron.Bias,
			})
		}
		state.Layers = append(state.Layers, layerState)
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(state)
}

func calculateError(patterns []Pattern) float64 {
	var totalError float64
	for _, pattern := range patterns {
		outputs, err := Execute(&mlp, &pattern)
		if err != nil {
			continue
		}
		for i, output := range outputs {
			diff := pattern.MultipleExpectation[i] - output
			totalError += diff * diff
		}
	}
	return totalError / float64(len(patterns))
}

func trainNetwork(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()

	var req TrainingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if len(req.Patterns) == 0 {
		http.Error(w, "No training patterns provided", http.StatusBadRequest)
		return
	}

	// Validate input dimensions
	inputSize := len(req.Patterns[0].Features)
	outputSize := len(req.Patterns[0].MultipleExpectation)
	
	if len(mlp.NeuralLayers[0].NeuronUnits) != inputSize {
		http.Error(w, fmt.Sprintf("Input size mismatch. Expected %d, got %d", len(mlp.NeuralLayers[0].NeuronUnits), inputSize), http.StatusBadRequest)
		return
	}

	if len(mlp.NeuralLayers[len(mlp.NeuralLayers)-1].NeuronUnits) != outputSize {
		http.Error(w, fmt.Sprintf("Output size mismatch. Expected %d, got %d", len(mlp.NeuralLayers[len(mlp.NeuralLayers)-1].NeuronUnits), outputSize), http.StatusBadRequest)
		return
	}

	epochs := 1000
	if req.Epochs > 0 {
		epochs = req.Epochs
	}

	// Train the network and update error
	for i := 0; i < epochs; i++ {
		MLPTrain(&mlp, req.Patterns, 1)
		currentEpoch = i + 1
		trainingError = calculateError(req.Patterns)
		
		// Test the patterns after training
		for _, pattern := range req.Patterns {
			Execute(&mlp, &pattern)
		}
	}

	state := NetworkState{
		Error: trainingError,
		Epoch: currentEpoch,
	}
	
	for _, layer := range mlp.NeuralLayers {
		layerState := LayerState{}
		for _, neuron := range layer.NeuronUnits {
			layerState.Neurons = append(layerState.Neurons, NeuronState{
				Weights: neuron.Weights,
				Value:   neuron.Value,
				Bias:    neuron.Bias,
			})
		}
		state.Layers = append(state.Layers, layerState)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(state)
}

func resetNetwork(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()

	mlp = PrepareMLPNet([]int{2, 4, 1}, 0.5, sigmoid, sigmoidDerivative)
	currentEpoch = 0
	trainingError = 0
	
	state := NetworkState{
		Error: trainingError,
		Epoch: currentEpoch,
	}
	
	for _, layer := range mlp.NeuralLayers {
		layerState := LayerState{}
		for _, neuron := range layer.NeuronUnits {
			layerState.Neurons = append(layerState.Neurons, NeuronState{
				Weights: neuron.Weights,
				Value:   neuron.Value,
				Bias:    neuron.Bias,
			})
		}
		state.Layers = append(state.Layers, layerState)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(state)
}

func main() {
	rand.Seed(time.Now().UnixNano())
	
	// Initialize MLP with 2 inputs, 4 hidden neurons, and 1 output
	mlp = PrepareMLPNet([]int{2, 4, 1}, 0.5, sigmoid, sigmoidDerivative)

	r := mux.NewRouter()
	r.HandleFunc("/api/network/state", getNetworkState).Methods("GET")
	r.HandleFunc("/api/network/train", trainNetwork).Methods("POST")
	r.HandleFunc("/api/network/reset", resetNetwork).Methods("POST")

	// Configure CORS
	c := cors.New(cors.Options{
		AllowedOrigins: []string{"http://localhost:3000"},
		AllowedMethods: []string{"GET", "POST", "OPTIONS"},
		AllowedHeaders: []string{"Content-Type"},
	})

	handler := c.Handler(r)
	fmt.Println("Server starting on :8080...")
	log.Fatal(http.ListenAndServe(":8080", handler))
}
