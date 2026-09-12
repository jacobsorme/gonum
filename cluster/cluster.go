// Copyright ©2020 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cluster

import (
	"math"
	"math/rand/v2"
)

// Generic implementation of KMeans. Use KMeans2D or KMeans3D for float64 2D or 3D float64 data
func Kmeans[T1 any](k int, data []T1, seed *uint64, epsilon float64, iter int, centroid func([]T1) T1, dist func(T1, T1) float64) (centroids []T1, clusters [][]T1) {
	centroids = make([]T1, k)
	random := make([]T1, max(len(data), k))
	copy(random, data)
	shuffleFunc := func(i, j int) {
		random[i], random[j] = random[j], random[i]
	}
	if seed != nil {
		rand.New(rand.NewPCG(*seed, *seed)).Shuffle(len(data), shuffleFunc)
	} else {
		rand.Shuffle(len(data), shuffleFunc)
	}

	copy(centroids, random[:k])

	calcClusters := func() {
		clusters = make([][]T1, k)
		for i := range data {
			d := data[i]
			closestIndex, minDistance := 0, dist(d, centroids[0])
			for j := range k {
				dist := dist(d, centroids[j])
				if dist < minDistance {
					closestIndex, minDistance = j, dist
				}
			}
			clusters[closestIndex] = append(clusters[closestIndex], d)
		}
	}
	if iter == 0 {
		calcClusters()
		return centroids, clusters
	}
	for range iter {
		calcClusters()
		allEqual := true
		newCentroids := make([]T1, k)
		for i := range k {
			newCentroid := centroid(clusters[i])
			allEqual = allEqual && dist(newCentroid, centroids[i]) <= epsilon
			newCentroids[i] = newCentroid
		}
		if allEqual {
			break
		}
		centroids = newCentroids
	}
	return centroids, clusters
}

type Config struct {
	Seed *uint64
	Iter *int
	Eps  float64
}

// KMeans for [x,y] float64 data points. Default config run up to 5 iterations.
// Tolerance for centroid equality is default 0.01, with option epsilon in config
// Centroid initialization defaults to rand.Shuffle, with seed options in config
// Setting 0 iterations returns the initialized centroids and their clusters
func Kmeans2D(k int, data [][2]float64, config Config) (centroids [][2]float64, clusters [][][2]float64) {
	dist := func(p1, p2 [2]float64) float64 {
		return math.Hypot(p1[0]-p2[0], p1[1]-p2[1])
	}
	centroid := func(cluster [][2]float64) [2]float64 {
		x, y := 0.0, 0.0
		for _, p := range cluster {
			x += p[0]
			y += p[1]
		}
		l := float64(len(cluster))
		return [2]float64{x / l, y / l}
	}
	if config.Eps <= 0 {
		config.Eps = 0.01
	}
	if config.Iter == nil {
		*config.Iter = 5
	}
	return Kmeans(k, data, config.Seed, config.Eps, *config.Iter, centroid, dist)
}

// KMeans for [x,y,z] float64 data points. Default config run up to 5 iterations.
// Tolerance for centroid equality is default 0.01, with option epsilon in config
// Centroid initialization defaults to rand.Shuffle, with seed options in config
// Setting 0 iterations returns the initialized centroids and their clusters
func Kmeans3D(k int, data [][3]float64, config Config) (centroids [][3]float64, clusters [][][3]float64) {
	dist := func(p1, p2 [3]float64) float64 {
		return math.Sqrt(math.Pow(p1[0]-p2[0], 2) + math.Pow(p1[1]-p2[1], 2) + math.Pow(p1[2]-p2[2], 2))
	}
	centroid := func(cluster [][3]float64) [3]float64 {
		x, y, z := 0.0, 0.0, 0.0
		for _, p := range cluster {
			x += p[0]
			y += p[1]
			y += p[2]
		}
		l := float64(len(cluster))
		return [3]float64{x / l, y / l, z / l}
	}
	if config.Eps <= 0 {
		config.Eps = 0.01
	}
	if config.Iter == nil {
		*config.Iter = 5
	}
	return Kmeans(k, data, config.Seed, config.Eps, *config.Iter, centroid, dist)
}
