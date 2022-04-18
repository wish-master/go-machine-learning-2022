package main

import (
	"bufio"
	"github.com/go-gota/gota/dataframe"
	"log"
	"os"
)

func main() {
	f, err := os.Open("/Users/wishmaster/Workstation/src/github.com/wish-master/go-machine-learning-2022/04-linear-regression/advertising.csv")
	if err != nil {
		log.Fatalf("can't open given input file: %v\n", err)
	}

	defer func() {
		_ = f.Close()
	}()

	df := dataframe.ReadCSV(f)

	trainingSetCount := df.Nrow() * 4 / 5
	testSetCount := df.Nrow() - trainingSetCount

	trainingSetIds := make([]int, trainingSetCount)
	testSetIds := make([]int, testSetCount)

	for i := 0; i < trainingSetCount; i++ {
		trainingSetIds[i] = i
	}

	for i := 0; i < testSetCount; i++ {
		testSetIds[i] = trainingSetCount + i
	}

	subset := df.Subset(trainingSetIds)
	exportDataframe("training.csv", subset)

	subset = df.Subset(testSetIds)
	exportDataframe("test.csv", subset)
}

func exportDataframe(filename string, df dataframe.DataFrame) {
	f, err := os.Create(filename)
	if err != nil {
		log.Fatalf("can't create given input file: %v\n", err)
	}

	w := bufio.NewWriter(f)
	err = df.WriteCSV(w)
	if err != nil {
		log.Fatalf("can't write given data: %v\n", err)
	}
}
