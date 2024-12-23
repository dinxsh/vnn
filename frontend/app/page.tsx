'use client';

import { useEffect, useRef, useState } from 'react';

interface NeuronState {
  weights: number[];
  value: number;
  bias: number;
}

interface LayerState {
  neurons: NeuronState[];
}

interface NetworkState {
  layers: LayerState[];
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [networkState, setNetworkState] = useState<NetworkState | null>(null);

  const fetchNetworkState = async () => {
    try {
      const response = await fetch('http://localhost:8080/api/network/state');
      const data = await response.json();
      setNetworkState(data);
    } catch (error) {
      console.error('Error fetching network state:', error);
    }
  };

  const trainNetwork = async () => {
    try {
      await fetch('http://localhost:8080/api/network/train', { method: 'POST' });
      await fetchNetworkState();
    } catch (error) {
      console.error('Error training network:', error);
    }
  };

  useEffect(() => {
    fetchNetworkState();
    const interval = setInterval(fetchNetworkState, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!canvasRef.current || !networkState) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const layerSpacing = canvas.width / (networkState.layers.length + 1);
    const maxNeurons = Math.max(...networkState.layers.map(l => l.neurons.length));
    const neuronSpacing = canvas.height / (maxNeurons + 1);
    const neuronRadius = 20;

    // Draw neurons and connections
    networkState.layers.forEach((layer, layerIndex) => {
      const x = layerSpacing * (layerIndex + 1);
      
      layer.neurons.forEach((neuron, neuronIndex) => {
        const y = (neuronSpacing * (neuronIndex + 1)) + 
                 ((canvas.height - (layer.neurons.length * neuronSpacing)) / 2);

        // Draw connections to previous layer
        if (layerIndex > 0) {
          const prevLayer = networkState.layers[layerIndex - 1];
          const prevX = layerSpacing * layerIndex;
          
          prevLayer.neurons.forEach((prevNeuron, prevIndex) => {
            const prevY = (neuronSpacing * (prevIndex + 1)) + 
                         ((canvas.height - (prevLayer.neurons.length * neuronSpacing)) / 2);
            
            // Draw connection with weight-based opacity
            const weight = Math.abs(neuron.weights[prevIndex]);
            ctx.beginPath();
            ctx.strokeStyle = `rgba(0, 0, 0, ${weight})`;
            ctx.moveTo(prevX, prevY);
            ctx.lineTo(x, y);
            ctx.stroke();
          });
        }

        // Draw neuron
        ctx.beginPath();
        ctx.fillStyle = `rgba(0, 123, 255, ${neuron.value})`;
        ctx.arc(x, y, neuronRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Draw neuron value
        ctx.fillStyle = 'black';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(neuron.value.toFixed(2), x, y);
      });
    });
  }, [networkState]);

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm">
        <h1 className="text-4xl font-bold mb-8 text-center">Neural Network Visualizer</h1>
        
        <div className="flex flex-col items-center gap-8">
          <canvas
            ref={canvasRef}
            width={800}
            height={600}
            className="border border-gray-300 rounded-lg"
          />
          
          <button
            onClick={trainNetwork}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
          >
            Train Network
          </button>
        </div>
      </div>
    </main>
  );
}
