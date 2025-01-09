package nn

import (
	"gotorch/tensor"
	"math"
	"math/rand"
)

type Dense struct {
	InputDim  int
	OutputDim int
	Weights   *tensor.Tensor
	Biases    *tensor.Tensor
	Input     *tensor.Tensor // Store for backprop

	// Gradients
	WeightGrads *tensor.Tensor
	BiasGrads   *tensor.Tensor
}

func NewDense(inputDim, outputDim int) *Dense {
	// Xavier/Glorot initialization
	scale := math.Sqrt(2.0 / float64(inputDim+outputDim))

	weights := make([]float64, inputDim*outputDim)
	for i := range weights {
		weights[i] = rand.NormFloat64() * scale
	}

	biases := make([]float64, outputDim)

	return &Dense{
		InputDim:    inputDim,
		OutputDim:   outputDim,
		Weights:     tensor.NewTensor(weights, inputDim, outputDim),
		Biases:      tensor.NewTensor(biases, outputDim),
		WeightGrads: tensor.NewTensor(make([]float64, inputDim*outputDim), inputDim, outputDim),
		BiasGrads:   tensor.NewTensor(make([]float64, outputDim), outputDim),
	}
}

func (d *Dense) Forward(input *tensor.Tensor) *tensor.Tensor {
	d.Input = input // Save for backprop

	// Compute output = input * weights + biases
	output, err := tensor.Dot(input, d.Weights)
	if err != nil {
		panic(err)
	}

	// Broadcast biases to match output shape
	batchSize := input.Shape[0]
	broadcastBiases := tensor.NewTensor(make([]float64, batchSize*d.OutputDim), batchSize, d.OutputDim)
	for i := 0; i < batchSize; i++ {
		copy(broadcastBiases.Data[i*d.OutputDim:(i+1)*d.OutputDim], d.Biases.Data)
	}

	result, err := tensor.Add(output, broadcastBiases)
	if err != nil {
		panic(err)
	}

	return result
}

func (d *Dense) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	// Compute gradients
	// dW = input^T * gradOutput
	inputT := tensor.Transpose(d.Input)
	d.WeightGrads, _ = tensor.Dot(inputT, gradOutput)

	// dB = sum(gradOutput, axis=0)
	d.BiasGrads = tensor.SumAxis(gradOutput, 0)

	// Compute gradient with respect to input
	// dX = gradOutput * W^T
	weightsT := tensor.Transpose(d.Weights)
	inputGrad, _ := tensor.Dot(gradOutput, weightsT)

	return inputGrad
}

func (d *Dense) UpdateParams(learningRate float64) {
	// Update weights and biases using gradients
	weightUpdate, _ := tensor.Multiply(d.WeightGrads, tensor.NewTensor(learningRate))
	d.Weights, _ = tensor.Subtract(d.Weights, weightUpdate)

	biasUpdate, _ := tensor.Multiply(d.BiasGrads, tensor.NewTensor(learningRate))
	d.Biases, _ = tensor.Subtract(d.Biases, biasUpdate)
}
