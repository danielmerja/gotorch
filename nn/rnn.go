package nn

import (
	af "gotorch/activation_functions"
	"gotorch/tensor"
	"math"
	"math/rand"
)

type RNN struct {
	InputDim  int
	HiddenDim int

	Wxh *tensor.Tensor // Input to hidden weights
	Whh *tensor.Tensor // Hidden to hidden weights
	Bh  *tensor.Tensor // Hidden bias

	States []*tensor.Tensor // Store hidden states for backprop
	Inputs []*tensor.Tensor // Store inputs for backprop
}

func NewRNN(inputDim, hiddenDim int) *RNN {
	scale := math.Sqrt(2.0 / float64(inputDim+hiddenDim))

	wxh := make([]float64, inputDim*hiddenDim)
	whh := make([]float64, hiddenDim*hiddenDim)
	bh := make([]float64, hiddenDim)

	for i := range wxh {
		wxh[i] = rand.NormFloat64() * scale
	}
	for i := range whh {
		whh[i] = rand.NormFloat64() * scale
	}

	return &RNN{
		InputDim:  inputDim,
		HiddenDim: hiddenDim,
		Wxh:       tensor.NewTensor(wxh, inputDim, hiddenDim),
		Whh:       tensor.NewTensor(whh, hiddenDim, hiddenDim),
		Bh:        tensor.NewTensor(bh, hiddenDim),
	}
}

func (r *RNN) Forward(input *tensor.Tensor) *tensor.Tensor {
	batchSize := input.Shape[0]
	seqLen := input.Shape[1]

	r.States = make([]*tensor.Tensor, seqLen+1)
	r.Inputs = make([]*tensor.Tensor, seqLen)
	r.States[0] = tensor.NewTensor([][]float64{
		make([]float64, r.HiddenDim),
	}, batchSize, r.HiddenDim)

	// For each time step
	for t := 0; t < seqLen; t++ {
		// Store input
		r.Inputs[t] = tensor.SliceTime(input, t)

		// Calculate hidden state: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
		wxh_x, err := tensor.Dot(r.Inputs[t], r.Wxh)
		if err != nil {
			panic(err)
		}
		whh_h, err := tensor.Dot(r.States[t], r.Whh)
		if err != nil {
			panic(err)
		}
		sum, err := tensor.Add(wxh_x, whh_h)
		if err != nil {
			panic(err)
		}
		sum, err = tensor.Add(sum, r.Bh)
		if err != nil {
			panic(err)
		}
		r.States[t+1] = af.Tanh(sum)
	}

	return r.States[len(r.States)-1]
}

// ... Backward and UpdateParams methods
