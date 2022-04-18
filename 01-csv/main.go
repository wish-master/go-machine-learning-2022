package main

import (
	"fmt"
	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/gonum/stat"
	"log"
	"os"
)

func main() {
	f, err := os.Open("/Users/wishmaster/Workstation/src/github.com/wish-master/go-machine-learning-2022/01-csv/simple.csv")
	if err != nil {
		log.Fatalf("can't open given input file: %v\n", err)
	}

	defer func() {
		_ = f.Close()
	}()

	df := dataframe.ReadCSV(f)
	fmt.Println(df)

	colForMeasure := df.Col("Number")

	modeVal, modeCount := stat.Mode(colForMeasure.Float(), nil)
	fmt.Printf("Mean: %f, Median: %f, Mode value: %f, Mode count: %f\n\n",
		colForMeasure.Mean(),
		colForMeasure.Median(),
		modeVal, modeCount)

	fValues := colForMeasure.Float()
	fmt.Printf("25 quantile: %f\n", stat.Quantile(.25, stat.Empirical, fValues, nil))
	fmt.Printf("50 quantile: %f\n", stat.Quantile(.5, stat.Empirical, fValues, nil))
	fmt.Printf("75 quantile: %f\n\n", stat.Quantile(.75, stat.Empirical, fValues, nil))

	fmt.Println(df.Describe())
}
