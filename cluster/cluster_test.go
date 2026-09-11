package cluster

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"strconv"
	"testing"
)

func TestKmeans2DZeroIter(t *testing.T) {
	data := [][2]float64{{rand.NormFloat64()*1 + 3, rand.NormFloat64()*1 + 3}}
	iter := 0
	cent, clust := Kmeans2D(4, data, Config{Iter: &iter})
	if len(cent) != 4 || len(clust) != 4 {
		t.Error("kmeans invalid dimensions")
	}
}

func TestKmeans2DEmptyData(t *testing.T) {
	data := [][2]float64{}
	iter := 10
	cent, clust := Kmeans2D(3, data, Config{Iter: &iter})
	if len(cent) != 3 || len(clust) != 3 {
		t.Error("kmeans invalid dimensions")
	}
}

func TestKmeans2D(t *testing.T) {
	data := [][2]float64{}
	for range 500 {
		data = append(data,
			[2]float64{rand.NormFloat64()*1 + 3, rand.NormFloat64()*1 + 3},
			[2]float64{rand.NormFloat64()*1 + 3, rand.NormFloat64()*1 - 3},
			[2]float64{rand.NormFloat64()*1 - 3, rand.NormFloat64()*1 + 3},
		)
	}
	expectedCenters := [][2]float64{{3, 3}, {3, -3}, {-3, 3}}
	iter := 5
	cent, clust := Kmeans2D(len(expectedCenters), data, Config{Seed1: 123, Seed2: 1234, Iter: &iter})
	if len(cent) != len(clust) {
		t.Error("kmeans invalid dimensions")
	}
	eps := 0.1
	for _, c1 := range cent {
		found := false
		for _, c2 := range expectedCenters {
			if math.Abs(c1[0]-c2[0]) <= eps && math.Abs(c1[1]-c2[1]) <= eps {
				found = true
			}
		}
		if !found {
			t.Errorf("center %+v not correct", c1)
		}
	}
}

func TestKmeans2DVisual(t *testing.T) {
	if os.Getenv("VISUAL_TEST") == "1" {
		type point struct {
			X     float64 `json:"x"`
			Y     float64 `json:"y"`
			Color string  `json:"color"`
		}
		html := `<html><body><form>
			<b>k = %d</b><input type="range" min="1" max="20" name="k" value="%d" style="width:200px"><br>
			<b>c = %d</b><input type="range" min="1" max="20" name="c" value="%d" style="width:200px"><br>
			<b>i = %d</b><input type="range" min="1" max="20" name="i" value="%d" style="width:200px"><br>
			<input type="submit" value="Run">
		</form><canvas id="canvas" width="900" height="800"></canvas>
		<script>
			let h = 700; let w = 900; let zoom = 20;
			let ctx = document.getElementById("canvas").getContext("2d")
			ctx.fillRect(0,h/2,w,1); ctx.fillRect(w/2,0,1,h)
			let d = %s
			for(let i in d.clusters) {
				let [r,g,b] = [Math.random()*200+20, Math.random()*200+20, Math.random()*200+20]
				ctx.setFillColor("rgb("+r+","+g+","+b+")")
				for(let p of d.clusters[i]) ctx.fillRect((zoom*p[0])+w/2,(zoom*-p[1])+h/2,3,3)
			}
			ctx.setFillColor("#333")
			for(let i = 0; i < d.centroidIterations.length; i++) {
				let centroids = d.centroidIterations[i]
				for(let j in centroids) {
					let [x,y] = centroids[j]
					if(i > 0) {
						let [x_, y_] = d.centroidIterations[i-1][j]
						ctx.beginPath(); ctx.lineWidth = 2
						ctx.moveTo((zoom*x_)+w/2, (zoom*-y_)+h/2)
						ctx.lineTo((zoom*x)+w/2, (zoom*-y)+h/2)
						ctx.strokeStyle = "#333"; ctx.stroke()
					}
					let s = i == d.centroidIterations.length-1 ? 10 : 5
					ctx.fillRect((zoom*x)+w/2, (zoom*-y)+h/2, s, s)
				}
			}
		</script></html></body>`

		http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			q := r.URL.Query()
			k, err := strconv.Atoi(q.Get("k"))
			if err != nil {
				k = 3
			}
			clusters, err := strconv.Atoi(q.Get("c"))
			if err != nil {
				clusters = 3
			}
			iter, err := strconv.Atoi(q.Get("i"))
			if err != nil {
				iter = 3
			}
			data := [][2]float64{}
			for range clusters {
				m1, m2 := -15+rand.Float64()*30, -15+rand.Float64()*30
				std1, std2 := 1+rand.Float64()*3, 1+rand.Float64()*3
				for range 500 {
					data = append(data, [2]float64{rand.NormFloat64()*std1 + m1, rand.NormFloat64()*std2 + m2})
				}
			}
			clust := [][][2]float64{}
			centroidIterations := make([][][2]float64, iter)
			for i := range iter {
				cent, c := Kmeans2D(k, data, Config{Seed1: 12, Seed2: 123, Iter: &i})
				centroidIterations[i] = cent
				clust = c
			}

			d, _ := json.Marshal(struct {
				Data               [][2]float64   `json:"data"`
				Clusters           [][][2]float64 `json:"clusters"`
				CentroidIterations [][][2]float64 `json:"centroidIterations"`
			}{Data: data, Clusters: clust, CentroidIterations: centroidIterations})
			fmt.Fprintf(w, html, k, k, clusters, clusters, iter, iter, d)
		})
		http.ListenAndServe(":8080", nil)
	}
}
