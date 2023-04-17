#include "meshsampler.h"

#include <random>
#include <limits>
#include <algorithm>
#include <numeric>
#include <iostream>

#include <igl/volume.h>

MeshSampler::MeshSampler(const Eigen::MatrixXd& verts, const Eigen::MatrixXi& tets) : MeshSampler(verts, tets, std::vector<int>(0)) {

}

MeshSampler::MeshSampler(const Eigen::MatrixXd& verts, const Eigen::MatrixXi& tets, const std::vector<int>& landmarks) : verts(verts), tets(tets), sample_diameter(-1.), sampled_verts(landmarks) {
	Eigen::VectorXd vol;
	igl::volume(verts, tets, vol);
	this->volume = std::abs(vol.sum());
}

std::vector<int> MeshSampler::getSampledVerts() { return this->sampled_verts; }

std::vector<double> MeshSampler::computeNearestNeighboursForSamples(std::vector<int> samples) {
	int num_verts = this->verts.rows();

	// Store nearest neighbour distances
	std::vector<double> nearest_neighbour_dist(num_verts - 1);
	std::fill(nearest_neighbour_dist.begin(), nearest_neighbour_dist.end(), std::numeric_limits<double>::max());

	for (int i = 0; i < samples.size(); ++i) {
		int current_sample = samples[i];
		// update new nearest neighbour distances
		for (int j = 0; j < nearest_neighbour_dist.size(); ++j) {
			double dist_to_prev_sampled_vert = (this->verts.row(current_sample) - this->verts.row(j)).squaredNorm();
			if (nearest_neighbour_dist[j] > dist_to_prev_sampled_vert) {
				nearest_neighbour_dist[j] = dist_to_prev_sampled_vert;
			}
		}
	}

	return nearest_neighbour_dist;
}

// This version uses EUCLIDEAN DISTANCE 
std::vector<int> MeshSampler::farthestPointSampling(int num_samples) {
	int num_verts = this->verts.rows();

	// Initialize the random number generator
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<int> dist(0, num_verts - 1);

	//https://minibatchai.com/ai/2021/08/07/FPS.html
	// store the sampled vertices
	//std::vector<int> sampled_verts;
	this->sampled_verts.reserve(num_samples);

	// store the unsampled vertices
	std::vector<int> unsampled_verts(num_verts);
	std::iota(std::begin(unsampled_verts), std::end(unsampled_verts), 0);

	// store all nearest neighbour distances
	std::vector<double> nearest_neighbour_dist = computeNearestNeighboursForSamples(this->sampled_verts);

	// Pick random vertex and put in sampled vertex and remove from unsampled verts
	// if no landmarks/sampled verts already exist
	if (this->sampled_verts.empty()) {
		sampled_verts.push_back(dist(gen));
	}

	int num_init_sampled_verts = this->sampled_verts.size();
	for (int i = 0; i < num_init_sampled_verts; ++i) {
		unsampled_verts.erase(std::remove(unsampled_verts.begin(), unsampled_verts.end(), sampled_verts[i]), unsampled_verts.end());
		nearest_neighbour_dist.erase(nearest_neighbour_dist.begin() + sampled_verts[i]);
	}

	for (int i = 0; i < num_samples - num_init_sampled_verts; ++i) {
		int prev_sampled_vert = sampled_verts.back();

		// update new nearest neighbour distances
		for (int j = 0; j < nearest_neighbour_dist.size(); ++j) {
			int current_vert = unsampled_verts[j];
			double dist_to_prev_sampled_vert = (this->verts.row(prev_sampled_vert) - this->verts.row(current_vert)).squaredNorm();
			if (nearest_neighbour_dist[j] > dist_to_prev_sampled_vert) {
				nearest_neighbour_dist[j] = dist_to_prev_sampled_vert;
			}
		}

		// Find the largest nearest neighbour distance
		auto largest_dist_it = std::max_element(std::begin(nearest_neighbour_dist), std::end(nearest_neighbour_dist));
		int next_sampled_vert_idx = std::distance(std::begin(nearest_neighbour_dist), largest_dist_it);

		// Put this vertex in the sampled list and remove from unsampled etc
		sampled_verts.push_back(unsampled_verts[next_sampled_vert_idx]);
		unsampled_verts.erase(unsampled_verts.begin() + next_sampled_vert_idx);
		nearest_neighbour_dist.erase(nearest_neighbour_dist.begin() + next_sampled_vert_idx);
	}

	// Set SampleDiameter to next largest nearest neighbour distance
	auto largest_dist = std::max_element(std::begin(nearest_neighbour_dist), std::end(nearest_neighbour_dist));
	this->sample_diameter = *largest_dist;
	return sampled_verts;
}

Eigen::MatrixXd MeshSampler::getRadialBaseFunctions(std::vector<int>& sampled_verts, double r, bool partition_of_one) {

	int num_samples = sampled_verts.size();
	int num_verts = this->verts.rows();

	double a = (1. / std::pow(r, 4.));
	double b = -2. * (1. / (r * r));


	Eigen::MatrixXd base_functions(num_verts, num_samples);
	base_functions.setZero();

	//
	double eps = std::sqrt(-std::log(0.0001)) / r;

	for (int i = 0; i < num_samples; ++i) {
		Eigen::Vector3d sampled_vert = this->verts.row(sampled_verts[i]);

		// Compute all distances to current sampled vertex
		for (int j = 0; j < num_verts; ++j) {
			double distance = (sampled_vert - verts.row(j).transpose()).squaredNorm();

			// Gaussian base function value (Currently set to quartic polynomial from HRPD)
			double base_function_val = a * std::pow(distance, 4.) + b * (distance * distance) + 1;//std::exp(-(distance * eps * distance * eps));// 
			// Set to zero if small enough
			if (distance >= r) {
				base_function_val = 0.0;
			}
			base_functions(j, i) = base_function_val;
		}
	}

	if (partition_of_one) {
		for (int i = 0; i < num_verts; ++i) {
			double sum = base_functions.row(i).sum();
			if (sum < 1e-6) {
				std::cout << "Warning: a vertex isn't properly covered by any of the radial basis functions!" << std::endl;
				double highest_val = 0.0;
				int highest_val_idx = -1;
				for (int j = 0; j < num_samples; ++j) {
					if (std::abs(base_functions(i, j)) > highest_val) {
						highest_val = base_functions(i, j);
						highest_val_idx = j;
					}
				}
				base_functions(i, highest_val_idx) = 1.;
			}
			else {
				base_functions.row(i) /= sum;

			}
		}
	}
	return base_functions;
}

Eigen::MatrixXd MeshSampler::createSkinningSpace(int num_samples) {

	// First create the skinning weights
	std::vector<int> samples = this->farthestPointSampling(num_samples);
	// Implementation based on hyper-reduced projective dynamics code
	// Equation 10 heuristic parameter from The hierarchical subspace iteration method for Laplace-Beltrami Eigenproblems (Nasikun et al.)
	double sigma = 7;
	double pi = 3.14159265358979323846;
	double r = std::sqrt((sigma * this->volume) / (num_samples * pi));

	Eigen::MatrixXd weight_mat = this->getRadialBaseFunctions(samples, r, true);

	// Create the skinning space
	// NOTE: currently making skinning space 3 x numverts so that it can be easier multiplied with the hessian
	// otherwise the hessian will need to be changed
	Eigen::MatrixXd skinning_space(3 * this->verts.rows(), 12 * num_samples);
	skinning_space.setZero();

	for (int i = 0; i < this->verts.rows(); ++i) {
		for (int j = 0; j < num_samples; ++j) {
			// set the q * w
			for (int d = 0; d < this->verts.cols(); ++d) {
				skinning_space(3 * i, 12 * j + 3 * d) = this->verts(i, d) * weight_mat(i, j);
				skinning_space(3 * i + 1, 12 * j + 3 * d + 1) = this->verts(i, d) * weight_mat(i, j);
				skinning_space(3 * i + 2, 12 * j + 3 * d + 2) = this->verts(i, d) * weight_mat(i, j);
			}
			// set the w
			skinning_space(3 * i, 12 * j + 9) = weight_mat(i, j);
			skinning_space(3 * i + 1, 12 * j + 10) = weight_mat(i, j);
			skinning_space(3 * i + 2, 12 * j + 11) = weight_mat(i, j);
		}
	}

	return skinning_space;
}
