package tensor

import (
	"fmt"
	"reflect"
	"testing"
)

/*
Unit tests for the Tensor package
*/

func Test_NewTensorScalar(t *testing.T) {

	tensor := NewTensor(3.14)

	expectedShape := []int{1}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{3.14}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}

}

func Test_NewTensorVector(t *testing.T) {

	tensor := NewTensor([]float64{1, 2})

	expectedShape := []int{2}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{1, 2}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for vector: got %v, expected %v", tensor.Data, expectedData)
	}

}

func Test_NewTensorMatrix1x3(t *testing.T) {

	tensor := NewTensor([][]float64{{1, 2, 3}})

	// expect a 1x3 matrix
	expectedShape := []int{1, 3}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	// since we flatten the data, we expect a 1-dimensional slice
	expectedData := []float64{1, 2, 3}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorMatrix2x3(t *testing.T) {

	tensor := NewTensor([][]float64{{1, 2, 3}, {4, 5, 6}})

	// expect a 2x3 matrix
	expectedShape := []int{2, 3}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	// since we flatten the data, we expect a 1-dimensional slice
	expectedData := []float64{1, 2, 3, 4, 5, 6}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorMatrix4x2(t *testing.T) {

	tensor := NewTensor([][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}})

	// expect a 2x3 matrix
	expectedShape := []int{4, 2}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	// since we flatten the data, we expect a 1-dimensional slice
	expectedData := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorInt(t *testing.T) {
	tensor := NewTensor(3)

	expectedShape := []int{1}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{3}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorInt32(t *testing.T) {
	tensor := NewTensor(int32(3))

	expectedShape := []int{1}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{3}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorInt64(t *testing.T) {
	tensor := NewTensor(int64(3))

	expectedShape := []int{1}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{3}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorFloat64(t *testing.T) {
	tensor := NewTensor(float64(3.53))

	expectedShape := []int{1}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{3.53}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorFloat64Slice(t *testing.T) {
	tensor := NewTensor([]float64{3.53})

	expectedShape := []int{1}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{3.53}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorFloat64SliceSlice(t *testing.T) {
	tensor := NewTensor([][]float64{{3.53, 234.2}})

	expectedShape := []int{1, 2}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{3.53, 234.2}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorPanic(t *testing.T) {

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()

	_ = NewTensor("hello")

}

func Test_NewTensorFrom2DSliceZeroLength(t *testing.T) {
	tensor := newTensorFrom2DSlice([][]float64{})

	expectedShape := []int{0, 0}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	if len(tensor.Data) != 0 {
		t.Errorf("Data slice should be empty, got %v", tensor.Data)
	}

}

func Test_NewTensorFRom2DslicePanic(t *testing.T) {

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic for rows of different lengths")
		}
	}()

	// This should cause a panic due to rows of different lengths
	_ = newTensorFrom2DSlice([][]float64{{1, 2}, {3}})

}

func TestDimsTensorScalar(t *testing.T) {

	scalar := NewTensor(3)

	expected := 1

	if scalar.Dims() != expected {
		t.Errorf("Incorrect dimensions for scalar, got %d, expected: %d", scalar.Dims(), expected)
	}
}

func TestDimsTensorVector(t *testing.T) {

	vector := NewTensor([]float64{1, 2})

	expected := 1

	if vector.Dims() != expected {
		t.Errorf("Incorrect dimensions for vector, got %d, expected: %d", vector.Dims(), expected)
	}
}

func TestDimsTensorMatrix(t *testing.T) {

	matrix := NewTensor([][]float64{{1, 2, 3}, {4, 5, 6}})

	expected := 2

	if matrix.Dims() != expected {
		t.Errorf("Incorrect dimensions for matrix, got %d, expected: %d", matrix.Dims(), expected)
	}
}

func Test_AddError(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor([]float64{1, 2})

	_, err := Add(s1, s2)
	if err == nil {
		t.Errorf("Cannot add two tensors of different shapes and dimensions")
	}
}

func Test_AddTensorScalar(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor(4)

	result, err := Add(s1, s2)
	if err != nil {
		t.Errorf("unable to add two tensor")
	}

	expected := float64(6)
	if result.Data[0] != expected {
		t.Errorf("Unable to add scalars correctly")
	}

}

func Test_AddTensorVector(t *testing.T) {
	s1 := NewTensor([]float64{6, 3})
	s2 := NewTensor([]float64{4, 1})

	result, err := Add(s1, s2)
	if err != nil {
		t.Errorf("unable to add two tensor")
	}

	expected := []float64{10, 4}
	for i := range result.Data {
		if result.Data[i] != expected[i] {
			t.Errorf("incorrect addition,expected: %v, got: %v", expected[i], result.Data[i])
		}
	}
}

func Test_AddTensorMatrix(t *testing.T) {
	s1 := NewTensor([][]float64{{6, 3}, {2, 6}})
	s2 := NewTensor([][]float64{{4, 1}, {5, 9}})

	result, err := Add(s1, s2)
	if err != nil {
		t.Errorf("unable to add two tensor")
	}

	expected := []float64{10, 4, 7, 15} // Flattened 2x2 matrix
	if len(result.Data) != len(expected) {
		t.Errorf("resulting tensor has incorrect number of elements: got %v, want %v", len(result.Data), len(expected))
	}

	for i, v := range result.Data {
		if v != expected[i] {
			t.Errorf("incorrect addition at index %d, expected: %v, got: %v", i, expected[i], v)
		}
	}
}

func Test_SubtractError(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor([]float64{1, 2})

	_, err := Subtract(s1, s2)
	if err == nil {
		t.Errorf("Cannot subtract two tensors of different shapes and dimensions")
	}
}

func Test_SubtractTensorScalar(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor(4)

	result, err := Subtract(s1, s2)
	if err != nil {
		t.Errorf("unable to subtract two tensor")
	}

	expected := float64(-2)
	if result.Data[0] != expected {
		t.Errorf("Unable to subtract scalars correctly, expected: %v, got: %v", expected, result.Data[0])
	}

}

func Test_SubtractTensorVector(t *testing.T) {
	s1 := NewTensor([]float64{6, 3})
	s2 := NewTensor([]float64{4, 1})

	result, err := Subtract(s1, s2)
	if err != nil {
		t.Errorf("unable to subtract two tensor")
	}

	expected := []float64{2, 2}
	for i := range result.Data {
		if result.Data[i] != expected[i] {
			t.Errorf("incorrect subtraction, expected: %v, got: %v", expected[i], result.Data[i])
		}
	}
}

func Test_SubtractTensorMatrix(t *testing.T) {
	s1 := NewTensor([][]float64{{6, 3}, {2, 6}})
	s2 := NewTensor([][]float64{{4, 1}, {5, 9}})

	result, err := Subtract(s1, s2)
	if err != nil {
		t.Errorf("unable to subtract two tensor")
	}

	expected := []float64{2, 2, -3, -3} // Flattened 2x2 matrix
	if len(result.Data) != len(expected) {
		t.Errorf("resulting tensor has incorrect number of elements: got %v, want %v", len(result.Data), len(expected))
	}

	for i, v := range result.Data {
		if v != expected[i] {
			t.Errorf("incorrect subtraction at index %d, expected: %v, got: %v", i, expected[i], v)
		}
	}
}

func Test_MultiplyError(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor([]float64{1, 2})

	_, err := Multiply(s1, s2)
	if err == nil {
		t.Errorf("Cannot multiply two tensors of different shapes and dimensions")
	}
}

func Test_MultiplyTensorScalar(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor(4)

	result, err := Multiply(s1, s2)
	if err != nil {
		t.Errorf("unable to multiply two tensor")
	}

	expected := float64(8)
	if result.Data[0] != expected {
		t.Errorf("Unable to multiply scalars correctly, expected: %v, got: %v", expected, result.Data[0])
	}

}

func Test_MultiplyTensorVector(t *testing.T) {
	s1 := NewTensor([]float64{6, 3})
	s2 := NewTensor([]float64{4, 1})

	result, err := Multiply(s1, s2)
	if err != nil {
		t.Errorf("unable to multiply two tensor")
	}

	expected := []float64{24, 3}
	for i := range result.Data {
		if result.Data[i] != expected[i] {
			t.Errorf("incorrect multiplication, expected: %v, got: %v", expected[i], result.Data[i])
		}
	}
}

func Test_MultiplyTensorMatrix(t *testing.T) {
	s1 := NewTensor([][]float64{{6, 3}, {2, 6}})
	s2 := NewTensor([][]float64{{4, 1}, {5, 9}})

	result, err := Multiply(s1, s2)
	if err != nil {
		t.Errorf("unable to multiply two tensor")
	}

	expected := []float64{24, 3, 10, 54} // Flattened 2x2 matrix
	if len(result.Data) != len(expected) {
		t.Errorf("resulting tensor has incorrect number of elements: got %v, want %v", len(result.Data), len(expected))
	}

	for i, v := range result.Data {
		if v != expected[i] {
			t.Errorf("incorrect multiplication at index %d, expected: %v, got: %v", i, expected[i], v)
		}
	}
}

func TestDivideError(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor([]float64{1, 2})

	_, err := Divide(s1, s2)
	if err == nil {
		t.Errorf("Cannot divide two tensors of different shapes and dimensions")
	}
}

func Test_DivideTensorScalar(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor(4)

	result, err := Divide(s1, s2)
	if err != nil {
		t.Errorf("unable to divide two tensor")
	}

	expected := float64(0.5)
	if result.Data[0] != expected {
		t.Errorf("Unable to divide scalars correctly, expected: %v, got: %v", expected, result.Data[0])
	}

}

func Test_DivideTensorVector(t *testing.T) {
	s1 := NewTensor([]float64{8, 3})
	s2 := NewTensor([]float64{4, 1})

	result, err := Divide(s1, s2)
	if err != nil {
		t.Errorf("unable to multiply two tensor")
	}

	expected := []float64{2, 3}
	for i := range result.Data {
		if result.Data[i] != expected[i] {
			t.Errorf("incorrect division, expected: %v, got: %v", expected[i], result.Data[i])
		}
	}
}

func Test_DivideTensorMatrix(t *testing.T) {
	s1 := NewTensor([][]float64{{8, 3}, {10, 27}})
	s2 := NewTensor([][]float64{{4, 1}, {5, 9}})

	result, err := Divide(s1, s2)
	if err != nil {
		t.Errorf("unable to multiply two tensor")
	}

	expected := []float64{2, 3, 2, 3} // Flattened 2x2 matrix
	if len(result.Data) != len(expected) {
		t.Errorf("resulting tensor has incorrect number of elements: got %v, want %v", len(result.Data), len(expected))
	}

	for i, v := range result.Data {
		if v != expected[i] {
			t.Errorf("incorrect division at index %d, expected: %v, got: %v", i, expected[i], v)
		}
	}
}

func Test_RandomError(t *testing.T) {

	rows := 1
	columns := 0
	_, err := Rand(rows, columns)
	if err == nil {
		t.Errorf("Expected an error due to a row or column being 0, got: %d and %d", rows, columns)
	}
}

func Test_Random1x2Tensor(t *testing.T) {

	rows := 1
	columns := 2
	result, err := Rand(rows, columns)
	if err != nil {
		t.Errorf("Incorrect tensor construction, got: %d and %d", rows, columns)
	}

	expected := []float64{.23, .3}
	if len(result.Data) != len(expected) {
		t.Errorf("Resulting tensor has incorrect number of elements: got %v, want %v", len(result.Data), len(expected))
	}

	for i := range result.Data {
		if result.Data[i] > 1 {
			t.Errorf("Resulting value should be between 0 and 1 got: %v", result.Data[i])
		}
	}

}

func Test_Random3x2(t *testing.T) {

	rows := 3
	columns := 2
	result, err := Rand(rows, columns)
	if err != nil {
		t.Errorf("Incorrect tensor construction, got: %d and %d", rows, columns)
	}

	fmt.Println("result", result)

	expected := []float64{.5, .7, .4, .9, .1, .3}
	if len(result.Data) != len(expected) {
		t.Errorf("Resulting tensor has incorrect number of elements: got %v, want %v", len(result.Data), len(expected))
	}

	for i := range result.Data {
		if result.Data[i] > 1 {
			t.Errorf("Resulting value should be between 0 and 1 got: %v", result.Data[i])
		}
	}

}

func Test_FormatTensorVector(t *testing.T) {

	tensor := NewTensor([]float64{1, 2, 3}, 3)
	expected := "[1.0000,2.0000,3.0000]"

	result := FormatTensor(tensor)
	if result != expected {
		t.Errorf("Expected %s, got %s", expected, result)
	}
}

func Test_FormatTensorMatrix(t *testing.T) {

	tensor := NewTensor([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	expected := "[\n  [1.0000,2.0000,3.0000],\n  [4.0000,5.0000,6.0000]\n]"

	result := FormatTensor(tensor)
	if result != expected {
		t.Errorf("Expected %s, got %s", expected, result)
	}
}

func Test_FormatTensorDeepMatrix(t *testing.T) {
	tensor := NewTensor([]float64{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2)
	expected := "[\n  [\n    [1.0000,2.0000],\n    [3.0000,4.0000]],\n  [\n    [5.0000,6.0000],\n    [7.0000,8.0000]]]"

	result := FormatTensor(tensor)
	if result != expected {
		t.Errorf("Expected %s, got %s", expected, result)
	}
}

func Test_DotVectorVector(t *testing.T) {
	v1 := NewTensor([]float64{1, 2, 3})
	v2 := NewTensor([]float64{4, 5, 6})

	result, err := Dot(v1, v2)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	expected := 32.0 // 1*4 + 2*5 + 3*6
	if result.Data[0] != expected {
		t.Errorf("vector dot product incorrect, got: %v, want: %v", result.Data[0], expected)
	}
}

func Test_DotMatrixVector(t *testing.T) {
	m1 := NewTensor([][]float64{{1, 2}, {3, 4}})
	v1 := NewTensor([]float64{5, 6})

	result, err := Dot(m1, v1)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	expected := []float64{17, 39} // [1*5 + 2*6, 3*5 + 4*6]
	if !reflect.DeepEqual(result.Data, expected) {
		t.Errorf("matrix-vector dot product incorrect, got: %v, want: %v", result.Data, expected)
	}
}

func Test_DotMatrixMatrix(t *testing.T) {
	m1 := NewTensor([][]float64{{1, 2}, {3, 4}})
	m2 := NewTensor([][]float64{{5, 6}, {7, 8}})

	result, err := Dot(m1, m2)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	expected := []float64{19, 22, 43, 50} // [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
	if !reflect.DeepEqual(result.Data, expected) {
		t.Errorf("matrix-matrix dot product incorrect, got: %v, want: %v", result.Data, expected)
	}
}

func Test_Transpose2x3Matrix(t *testing.T) {
	tensor := NewTensor([][]float64{{1, 2, 3}, {4, 5, 6}})
	result := Transpose(tensor)

	expectedShape := []int{3, 2}
	expectedData := []float64{1, 4, 2, 5, 3, 6}

	if !reflect.DeepEqual(result.Shape, expectedShape) {
		t.Errorf("Incorrect shape after transpose: got %v, want %v", result.Shape, expectedShape)
	}
	if !reflect.DeepEqual(result.Data, expectedData) {
		t.Errorf("Incorrect data after transpose: got %v, want %v", result.Data, expectedData)
	}
}

func Test_TransposePanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for non-2D tensor")
		}
	}()

	tensor := NewTensor([]float64{1, 2, 3})
	Transpose(tensor)
}

func Test_SumAxis0Matrix(t *testing.T) {
	tensor := NewTensor([][]float64{{1, 2, 3}, {4, 5, 6}})
	result := SumAxis(tensor, 0)

	expectedShape := []int{3}
	expectedData := []float64{5, 7, 9}

	if !reflect.DeepEqual(result.Shape, expectedShape) {
		t.Errorf("Incorrect shape after sum: got %v, want %v", result.Shape, expectedShape)
	}
	if !reflect.DeepEqual(result.Data, expectedData) {
		t.Errorf("Incorrect data after sum: got %v, want %v", result.Data, expectedData)
	}
}

func Test_SumAxisPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for invalid axis")
		}
	}()

	tensor := NewTensor([][]float64{{1, 2}, {3, 4}})
	SumAxis(tensor, 2) // Should panic - axis out of bounds
}

func Test_FormatTensorInvalidShape(t *testing.T) {
	tensor := &Tensor{
		Data:  []float64{1, 2, 3},
		Shape: []int{}, // Invalid empty shape
	}

	result := FormatTensor(tensor)
	expected := "Unable to format tensor"
	if result != expected {
		t.Errorf("Expected %s for invalid tensor, got %s", expected, result)
	}
}

func Test_FormatTensorRecursive(t *testing.T) {
	// Test 3D tensor formatting
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	shape := []int{2, 2, 2}

	result := formatTensorRecursive(data, shape, 0)
	expected := "[\n  [\n    [1.0000,2.0000],\n    [3.0000,4.0000]],\n  [\n    [5.0000,6.0000],\n    [7.0000,8.0000]]]"

	if result != expected {
		t.Errorf("Incorrect recursive formatting:\nexpected:\n%s\ngot:\n%s", expected, result)
	}
}

func Test_FormatTensorRecursiveScalar(t *testing.T) {
	data := []float64{3.14159}
	shape := []int{}

	result := formatTensorRecursive(data, shape, 0)
	expected := "3.1416"

	if result != expected {
		t.Errorf("Incorrect scalar formatting: expected %s, got %s", expected, result)
	}
}
