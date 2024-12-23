package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"time"
	"github.com/gorilla/mux"
	"github.com/rs/cors"
)

type NetworkState struct {
	Layers []LayerState `json:"layers"`
}

type LayerState struct {
	Neurons []NeuronState `json:"neurons"`
}

type NeuronState struct {
	Weights []float64 `json:"weights"`
	Value   float64   `json:"value"`
	Bias    float64   `json:"bias"`
}

var mlp MultiLayerNetwork

func getNetworkState(w http.ResponseWriter, r *http.Request) {
	state := NetworkState{}
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
	json.NewEncoder(w).Encode(state)
}

func trainNetwork(w http.ResponseWriter, r *http.Request) {
	patterns := []Pattern{
		{Features: []float64{0, 0}, MultipleExpectation: []float64{0}},
		{Features: []float64{0, 1}, MultipleExpectation: []float64{1}},
		{Features: []float64{1, 0}, MultipleExpectation: []float64{1}},
		{Features: []float64{1, 1}, MultipleExpectation: []float64{0}},
	}
	MLPTrain(&mlp, patterns, 100)
	w.WriteHeader(http.StatusOK)
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Initialize MLP
	mlp = PrepareMLPNet([]int{2, 4, 1}, 0.5, sigmoid, sigmoidDerivative)

	r := mux.NewRouter()
	r.HandleFunc("/api/network/state", getNetworkState).Methods("GET")
	r.HandleFunc("/api/network/train", trainNetwork).Methods("POST")

	// Configure CORS
	c := cors.New(cors.Options{
		AllowedOrigins: []string{"http://localhost:3000"},
		AllowedMethods: []string{"GET", "POST", "OPTIONS"},
	})

	handler := c.Handler(r)
	fmt.Println("Server starting on :8080...")
	log.Fatal(http.ListenAndServe(":8080", handler))
}
