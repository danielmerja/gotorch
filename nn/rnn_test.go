package nn

import (
	"gotorch/tensor"
	"reflect"
	"testing"
)

func Test_NewRNN(t *testing.T) {
	inputDim := 4
	hiddenDim := 3

	rnn := NewRNN(inputDim, hiddenDim)

	// Check dimensions
	expectedWxhShape := []int{inputDim, hiddenDim}
	if !reflect.DeepEqual(rnn.Wxh.Shape, expectedWxhShape) {
		t.Errorf("Incorrect Wxh shape: got %v, want %v", rnn.Wxh.Shape, expectedWxhShape)
	}

	expectedWhhShape := []int{hiddenDim, hiddenDim}
	if !reflect.DeepEqual(rnn.Whh.Shape, expectedWhhShape) {
		t.Errorf("Incorrect Whh shape: got %v, want %v", rnn.Whh.Shape, expectedWhhShape)
	}

	expectedBhShape := []int{hiddenDim}
	if !reflect.DeepEqual(rnn.Bh.Shape, expectedBhShape) {
		t.Errorf("Incorrect bias shape: got %v, want %v", rnn.Bh.Shape, expectedBhShape)
	}
}

// test fails
func Test_RNNForward(t *testing.T) {
	inputDim := 2
	hiddenDim := 3
	rnn := NewRNN(inputDim, hiddenDim)

	// Set weights for predictable output
	rnn.Wxh = tensor.NewTensor([][]float64{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	})
	rnn.Whh = tensor.NewTensor([][]float64{
		{0.7, 0.8, 0.9},
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	})
	rnn.Bh = tensor.NewTensor([][]float64{{0.1, 0.2, 0.3}}, 1, hiddenDim)

	// Input: batch size of 2, sequence length of 2, input dim of 2
	input := tensor.NewTensor([]float64{
		0.0, 0.1, // batch 0, time 0
		0.2, 0.3, // batch 0, time 1
		0.4, 0.5, // batch 1, time 0
		0.6, 0.7, // batch 1, time 1
	}, 2, 2, 2) // [batch_size, seq_len, input_dim]

	for i := range input.Data {
		input.Data[i] = float64(i) * 0.1
	}

	output := rnn.Forward(input)

	// Check output dimensions
	expectedShape := []int{2, hiddenDim} // [batch_size, hidden_dim]
	if !reflect.DeepEqual(output.Shape, expectedShape) {
		t.Errorf("Incorrect output shape: got %v, want %v", output.Shape, expectedShape)
	}

	// Check states were stored correctly
	if len(rnn.States) != 3 { // seqLen + 1
		t.Errorf("Incorrect number of states stored: got %d, want %d", len(rnn.States), 3)
	}

	if len(rnn.Inputs) != 2 { // seqLen
		t.Errorf("Incorrect number of inputs stored: got %d, want %d", len(rnn.Inputs), 2)
	}
}

func Test_RNNForwardSingleTimeStep(t *testing.T) {
	inputDim := 2
	hiddenDim := 2
	rnn := NewRNN(inputDim, hiddenDim)

	// Set simple weights
	rnn.Wxh = tensor.NewTensor([][]float64{
		{1, 0},
		{0, 1},
	})
	rnn.Whh = tensor.NewTensor([][]float64{
		{0, 0},
		{0, 0},
	})
	rnn.Bh = tensor.NewTensor([][]float64{{0, 0}}, 1, hiddenDim)

	// Single time step input needs proper shape [batch_size, seq_len, input_dim]
	input := tensor.NewTensor([]float64{1, 2}, 1, 1, 2)

	output := rnn.Forward(input)

	expectedShape := []int{1, hiddenDim}
	if !reflect.DeepEqual(output.Shape, expectedShape) {
		t.Errorf("Incorrect output shape: got %v, want %v", output.Shape, expectedShape)
	}

	// With these weights and zero bias, output should be tanh of input
	expectedFirstValue := float64(1)
	expectedSecondValue := float64(2)
	if !floatEquals(output.Data[0], expectedFirstValue, 1e-6) {
		t.Errorf("Incorrect first output value: got %v, want %v", output.Data[0], expectedFirstValue)
	}
	if !floatEquals(output.Data[1], expectedSecondValue, 1e-6) {
		t.Errorf("Incorrect second output value: got %v, want %v", output.Data[1], expectedSecondValue)
	}
}
