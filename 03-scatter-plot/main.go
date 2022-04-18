package main

import (
	"fmt"
	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"image/color"
	"log"
	"os"
)

func main() {
	f, err := os.Open("/Users/wishmaster/Workstation/src/github.com/wish-master/go-machine-learning-2022/03-scatter-plot/advertising.csv")
	if err != nil {
		log.Fatalf("can't open given input file: %v\n", err)
	}

	defer func() {
		_ = f.Close()
	}()

	df := dataframe.ReadCSV(f)
	fmt.Println(df.Describe())

	pv := make(plotter.Values, df.Nrow())
	// histogram
	for _, colName := range df.Names() {
		if colName == "Sales" {
			continue
		}

		for i, tmp := range df.Col(colName).Float() {
			pv[i] = tmp
		}

		p := plot.New()
		p.Title.Text = "Histogram for " + colName

		h, err := plotter.NewHist(pv, 20)
		if err != nil {
			log.Fatalln(err)
		}

		h.FillColor = color.RGBA{R: 255, A: 255}
		h.Normalize(1)
		p.Add(h)

		_ = p.Save(10*vg.Centimeter, 10*vg.Centimeter, colName+".png")
	}

	// scatter plot
	yValues := df.Col("Sales").Float()
	points := make(plotter.XYs, df.Nrow())
	for _, colName := range df.Names() {
		for i, tmp := range df.Col(colName).Float() {
			points[i].X = tmp
			points[i].Y = yValues[i]
		}

		p := plot.New()
		p.Title.Text = "Scatter plot for " + colName
		p.X.Label.Text = colName
		p.Y.Label.Text = "Sales"

		s, err := plotter.NewScatter(points)
		if err != nil {
			log.Fatalln(err)
		}

		s.GlyphStyle.Color = color.RGBA{R: 255, A: 255}
		p.Add(s)
		p.Add(plotter.NewGrid())

		_ = p.Save(10*vg.Centimeter, 10*vg.Centimeter, colName+"_scatter.png")
	}
}
