package main

import (
	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"image/color"
	"log"
	"os"
)

func main() {
	f, err := os.Open("/Users/wishmaster/Workstation/src/github.com/wish-master/go-machine-learning-2022/02-plot/simple.csv")
	if err != nil {
		log.Fatalf("can't open given input file: %v\n", err)
	}

	defer func() {
		_ = f.Close()
	}()

	df := dataframe.ReadCSV(f)
	pv := make(plotter.Values, df.Nrow())
	for i, tmp := range df.Col("Number").Float() {
		pv[i] = tmp
	}

	p := plot.New()
	p.Title.Text = "Histogram for Number"

	h, err := plotter.NewHist(pv, 20)
	if err != nil {
		log.Fatalln(err)
	}

	h.FillColor = color.RGBA{R: 255, A: 255}
	p.Add(h)

	_ = p.Save(10*vg.Centimeter, 10*vg.Centimeter, "number.png")

	f, err = os.Open("/Users/wishmaster/Workstation/src/github.com/wish-master/go-machine-learning-2022/02-plot/iris.csv")
	if err != nil {
		log.Fatalf("can't open given input file: %v\n", err)
	}

	defer func() {
		_ = f.Close()
	}()

	df = dataframe.ReadCSV(f)
	pv = make(plotter.Values, df.Nrow())
	for _, col := range df.Names() {
		if col == "variety" {
			continue
		}

		for i, tmp := range df.Col(col).Float() {
			pv[i] = tmp
		}

		p := plot.New()
		p.Title.Text = "Histogram for " + col

		h, err := plotter.NewHist(pv, 20)
		if err != nil {
			log.Fatalln(err)
		}

		h.FillColor = color.RGBA{R: 255, A: 255}
		h.Normalize(1)
		p.Add(h)

		_ = p.Save(10*vg.Centimeter, 10*vg.Centimeter, col+".png")
	}
}
