package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"github.com/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

type networkConfig struct {
	inputCount    int
	outputCount   int
	hiddenCount   int
	numberOfSteps int
	rate          float64
}

type networkModel struct {
	config       networkConfig
	weightHidden *mat.Dense
	biasHidden   *mat.Dense
	weightOutput *mat.Dense
	biasOutput   *mat.Dense
}

func createNetwork(config networkConfig) *networkModel {
	return &networkModel{config, nil, nil, nil, nil}
}

func sigmoid(a float64) float64 {
	return 1. / (1. + math.Exp(-a))
}

func sigmoidPrime(a float64) float64 {
	return sigmoid(a) * (1. - sigmoid(a))
}

// sumAlongAxis sums a matrix along a particular dimension,
// preserving the other dimension.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

func (m *networkModel) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {

	// Loop over the number of epochs utilizing
	// backpropagation to train our model.
	for i := 0; i < m.config.numberOfSteps; i++ {

		// Complete the feed forward process.
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// Complete the backpropagation.
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// Adjust the parameters.
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(m.config.rate, wOutAdj) // RATE
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(m.config.rate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(m.config.rate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(m.config.rate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}

	return nil
}

func (m *networkModel) train(x, y *mat.Dense) error {
	rs := rand.NewSource(time.Now().UnixNano())
	rv := rand.New(rs)

	hiddenWeights := mat.NewDense(m.config.inputCount, m.config.hiddenCount, nil)
	hiddenBiases := mat.NewDense(1, m.config.hiddenCount, nil)
	outputWeights := mat.NewDense(m.config.hiddenCount, m.config.outputCount, nil)
	outputBiases := mat.NewDense(1, m.config.outputCount, nil)

	tmpCollection := [][]float64{
		hiddenWeights.RawMatrix().Data,
		hiddenBiases.RawMatrix().Data,
		outputWeights.RawMatrix().Data,
		outputBiases.RawMatrix().Data,
	}

	for i := range tmpCollection {
		for j := range tmpCollection[i] {
			tmpCollection[i][j] = rv.Float64()
		}
	}

	output := new(mat.Dense)
	err := m.backpropagate(x, y, hiddenWeights, hiddenBiases, outputWeights, outputBiases, output)
	if err != nil {
		return fmt.Errorf("error while backpropagate: %w", err)
	}

	m.weightHidden = hiddenWeights
	m.biasHidden = hiddenBiases
	m.weightOutput = outputWeights
	m.biasOutput = outputBiases

	return nil
}

// predict makes a prediction based on a trained
// neural network.
func (m *networkModel) predict(x *mat.Dense) (*mat.Dense, error) {

	// Check to make sure that our neuralNet value
	// represents a trained model.
	if m.weightHidden == nil || m.weightOutput == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if m.biasHidden == nil || m.biasOutput == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// Define the output of the neural network.
	output := new(mat.Dense)

	// Complete the feed forward process.
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, m.weightHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + m.biasHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, m.weightOutput)
	addBOut := func(_, col int, v float64) float64 { return v + m.biasOutput.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

func makeInputsAndLabels(fileName string) (*mat.Dense, *mat.Dense) {
	// Open the dataset file.
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		_ = f.Close()
	}()

	// Create a new CSV reader reading from the opened file.
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	// Read in all of the CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData and labelsData will hold all the
	// float values that will eventually be
	// used to form matrices.
	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	// Will track the current index of matrix values.
	var inputsIndex int
	var labelsIndex int

	// Sequentially move the rows into a slice of floats.
	for idx, record := range rawCSVData {

		// Skip the header row.
		if idx == 0 {
			continue
		}

		// Loop over the float columns.
		for i, val := range record {

			// Convert the value to a float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add to the labelsData if relevant.
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add the float value to the slice of floats.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)
	return inputs, labels
}

func main() {
	// Form the training matrices.
	inputs, labels := makeInputsAndLabels("train.csv")

	// Define our network architecture and learning parameters.
	config := networkConfig{
		inputCount:    4,
		outputCount:   3,
		hiddenCount:   3,
		numberOfSteps: 5000,
		rate:          0.3,
	}

	// Train the neural network.
	network := createNetwork(config)
	if err := network.train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	// Form the testing matrices.
	testInputs, testLabels := makeInputsAndLabels("test.csv")

	// Make the predictions using the trained model.
	predictions, err := network.predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate the accuracy of our model.
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {

		// Get the label.
		labelRow := mat.Row(nil, i, testLabels)
		var prediction int
		for idx, label := range labelRow {
			if label == 1.0 {
				prediction = idx
				break
			}
		}

		// Accumulate the true positive/negative count.
		if predictions.At(i, prediction) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	// Calculate the accuracy (subset accuracy).
	accuracy := float64(truePosNeg) / float64(numPreds)

	// Output the Accuracy value to standard out.
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)
}
