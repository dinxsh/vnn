## Multilayer Perceptron in Golang

a simple yet powerful neural network from scratch. Let's dive in!

### 🚀 Overview

An MLP is a type of artificial neural network (ANN) that consists of an input layer, one or more hidden layers, and an output layer. Each neuron in one layer is connected to every neuron in the next layer, making it a fully connected network.

### 🧠 Key Components

- **Pattern**: Represents the input data and expected output.
- **NeuronUnit**: The building block of the network, with weights, bias, and more.
- **NeuralLayer**: A collection of neurons forming a layer.
- **MultiLayerNetwork**: The entire network, ready to learn and predict!

### 🔧 Core Functions

- **PrepareLayer**: Sets up a layer with neurons.
- **PrepareMLPNet**: Initializes the MLP with layers and learning parameters.
- **Execute**: Runs the network to get predictions.
- **BackPropagate**: Adjusts weights using backpropagation.
- **MLPTrain**: Trains the network with data over multiple epochs.

### 📈 Transfer Functions

- **Sigmoid Function**: Adds non-linearity to the model.
- **Sigmoid Derivative**: Helps in calculating gradients during training.

## 🎮 How to Use

1. **Define Patterns**: Set up your input data and expected results.
2. **Initialize MLP**: Use `PrepareMLPNet` to create your network.
3. **Train**: Call `MLPTrain` to teach the network.
4. **Test**: Use `Execute` to see how well it learned!

## 🧪 Example

Check out the included example that tackles the classic XOR problem. It's a great way to see the MLP in action!

## 📚 Learn More

For a deep dive into the theory and code, visit the [original article](https://madeddu.xyz/posts/neuralnetwork/).

## 🤝 Contributing

Feel free to fork this repo, make improvements, and submit pull requests. Let's make this project even better together!

## 📜 License

This project is open-source under the MIT License. Enjoy and happy coding! 🎉

---

Thanks for checking out this project! If you have any questions or feedback, don't hesitate to reach out. Let's build something amazing! 🚀
