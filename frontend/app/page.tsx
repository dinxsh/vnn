'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

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
  error: number;
  epoch: number;
}

interface TrainingPattern {
  features: number[];
  multipleExpectation: number[];
}

const DEFAULT_PATTERNS = [
  { features: [0, 0], multipleExpectation: [0] },
  { features: [0, 1], multipleExpectation: [1] },
  { features: [1, 0], multipleExpectation: [1] },
  { features: [1, 1], multipleExpectation: [0] },
];

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [networkState, setNetworkState] = useState<NetworkState | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [epochs, setEpochs] = useState(1000);
  const [patterns, setPatterns] = useState<TrainingPattern[]>(DEFAULT_PATTERNS);
  const [newPattern, setNewPattern] = useState<TrainingPattern>({
    features: [0, 0],
    multipleExpectation: [0],
  });
  const [mounted, setMounted] = useState(false);

  const drawNetwork = useCallback(() => {
    if (!canvasRef.current || !networkState) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const layerSpacing = canvas.width / (networkState.layers.length + 1);
    const maxNeurons = Math.max(...networkState.layers.map(l => l.neurons.length));
    const neuronSpacing = canvas.height / (maxNeurons + 1);
    const neuronRadius = 15;

    // Draw connections first
    networkState.layers.forEach((layer, layerIndex) => {
      if (layerIndex === 0) return; // Skip input layer connections

      const x = layerSpacing * (layerIndex + 1);
      layer.neurons.forEach((neuron, neuronIndex) => {
        const y = (neuronSpacing * (neuronIndex + 1)) +
          ((canvas.height - (layer.neurons.length * neuronSpacing)) / 2);

        // Draw connections to previous layer
        const prevLayer = networkState.layers[layerIndex - 1];
        const prevX = layerSpacing * layerIndex;
        prevLayer.neurons.forEach((prevNeuron, prevIndex) => {
          const prevY = (neuronSpacing * (prevIndex + 1)) +
            ((canvas.height - (prevLayer.neurons.length * neuronSpacing)) / 2);
          
          const weight = neuron.weights[prevIndex];
          const alpha = Math.abs(weight);
          ctx.strokeStyle = weight > 0 ? `rgba(0, 255, 0, ${alpha})` : `rgba(255, 0, 0, ${alpha})`;
          ctx.lineWidth = Math.abs(weight) * 2;
          
          ctx.beginPath();
          ctx.moveTo(prevX, prevY);
          ctx.lineTo(x, y);
          ctx.stroke();
        });
      });
    });

    // Draw neurons
    networkState.layers.forEach((layer, layerIndex) => {
      const x = layerSpacing * (layerIndex + 1);
      layer.neurons.forEach((neuron, neuronIndex) => {
        const y = (neuronSpacing * (neuronIndex + 1)) +
          ((canvas.height - (layer.neurons.length * neuronSpacing)) / 2);

        ctx.beginPath();
        ctx.arc(x, y, neuronRadius, 0, Math.PI * 2);
        ctx.fillStyle = `rgb(
          ${Math.floor(255 * neuron.value)},
          ${Math.floor(255 * neuron.value)},
          ${Math.floor(255 * neuron.value)}
        )`;
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw bias
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(neuron.bias.toFixed(2), x, y + neuronRadius + 15);
      });
    });
  }, [networkState, canvasRef]);

  const fetchNetworkState = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8080/api/network/state');
      const data = await response.json();
      setNetworkState(data);
      drawNetwork();
    } catch (error) {
      console.error('Error fetching network state:', error);
    }
  }, [drawNetwork]);

  const trainNetwork = useCallback(async () => {
    try {
      setIsTraining(true);
      const response = await fetch('http://localhost:8080/api/network/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ patterns, epochs }),
      });
      const data = await response.json();
      setNetworkState(data);
      drawNetwork();
    } catch (error) {
      console.error('Error training network:', error);
    } finally {
      setIsTraining(false);
    }
  }, [patterns, epochs, drawNetwork]);

  const resetNetwork = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8080/api/network/reset', {
        method: 'POST',
      });
      const data = await response.json();
      setNetworkState(data);
      drawNetwork();
    } catch (error) {
      console.error('Error resetting network:', error);
    }
  }, [drawNetwork]);

  const addPattern = useCallback(() => {
    setPatterns(prev => [...prev, { ...newPattern }]);
    setNewPattern({
      features: [0, 0],
      multipleExpectation: [0],
    });
  }, [newPattern]);

  const removePattern = useCallback((index: number) => {
    setPatterns(prev => prev.filter((_, i) => i !== index));
  }, []);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;
    
    fetchNetworkState();
    const interval = setInterval(fetchNetworkState, 1000);
    return () => clearInterval(interval);
  }, [mounted, fetchNetworkState]);

  useEffect(() => {
    if (!mounted) return;
    drawNetwork();
  }, [mounted, drawNetwork]);

  if (!mounted) {
    return null;
  }

  return (
    <main className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold mb-8 text-center bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          Neural Network Visualizer
        </h1>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Training Data Panel */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-gray-800 p-6 rounded-xl shadow-lg">
              <h2 className="text-xl font-semibold mb-4">Training Settings</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Epochs</label>
                  <input
                    type="number"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value) || 1000)}
                    className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    min="1"
                    max="10000"
                  />
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={trainNetwork}
                    disabled={isTraining}
                    className={`flex-1 px-4 py-2 ${
                      isTraining 
                        ? 'bg-gray-600 cursor-not-allowed' 
                        : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700'
                    } rounded-md transition-all transform hover:scale-105`}
                  >
                    {isTraining ? 'Training...' : 'Train Network'}
                  </button>
                  <button
                    onClick={resetNetwork}
                    disabled={isTraining}
                    className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-md transition-colors"
                  >
                    Reset
                  </button>
                </div>
              </div>
            </div>

            <div className="bg-gray-800 p-6 rounded-xl shadow-lg">
              <h2 className="text-xl font-semibold mb-4">Training Data</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Input Features</label>
                  <input
                    type="text"
                    value={newPattern.features.join(', ')}
                    onChange={(e) => setNewPattern({
                      ...newPattern,
                      features: e.target.value.split(',').map(n => Number(n) || 0),
                    })}
                    className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    placeholder="0, 0"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">Expected Output</label>
                  <input
                    type="text"
                    value={newPattern.multipleExpectation.join(', ')}
                    onChange={(e) => setNewPattern({
                      ...newPattern,
                      multipleExpectation: e.target.value.split(',').map(n => Number(n) || 0),
                    })}
                    className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    placeholder="0"
                  />
                </div>
                
                <button
                  onClick={addPattern}
                  className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-md transition-colors"
                >
                  Add Pattern
                </button>
              </div>
            </div>

            {/* Pattern List */}
            <div className="bg-gray-800 p-6 rounded-xl shadow-lg">
              <h3 className="text-lg font-medium mb-3">Training Patterns</h3>
              <div className="space-y-3">
                {patterns.map((pattern, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-700 rounded">
                    <div className="text-sm">
                      <span className="text-gray-300">Input: </span>
                      <span className="font-mono">[{pattern.features.join(', ')}]</span>
                      <span className="text-gray-300 ml-2">→</span>
                      <span className="font-mono ml-2">[{pattern.multipleExpectation.join(', ')}]</span>
                    </div>
                    <button
                      onClick={() => removePattern(index)}
                      className="p-1 text-red-400 hover:text-red-300"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {/* Network Visualization */}
          <div className="lg:col-span-2 bg-gray-800 p-6 rounded-xl shadow-lg">
            <div className="mb-4 flex justify-between items-center">
              <div>
                <span className="text-gray-400">Epoch: </span>
                <span className="font-mono">{networkState?.epoch || 0}</span>
              </div>
              <div>
                <span className="text-gray-400">Error: </span>
                <span className="font-mono">{networkState?.error?.toFixed(6) || 0}</span>
              </div>
            </div>
            
            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              className="w-full h-[600px] rounded-lg"
            />
            
            <div className="mt-4 text-sm text-gray-400">
              <div>• Connection colors: Green (positive weight), Red (negative weight)</div>
              <div>• Connection opacity and thickness indicate weight strength</div>
              <div>• Neuron brightness indicates activation level</div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
