#ifndef MESH_SAMPLER_H
#define MESH_SAMPLER_H

#include <Eigen/dense>


class MeshSampler {
private:
	double sample_diameter;
	std::vector<int> sampled_verts;
	Eigen::MatrixXd verts;
	Eigen::MatrixXi tets;
	double volume;
public:
	MeshSampler(const Eigen::MatrixXd& verts, const Eigen::MatrixXi& tets);
	MeshSampler(const Eigen::MatrixXd& verts, const Eigen::MatrixXi& tets, const std::vector<int>& landmarks);

	std::vector<int> getSampledVerts();

	std::vector<double> computeNearestNeighboursForSamples(std::vector<int> samples);

	// Currently dont need tris and tets
	// This version uses EUCLIDEAN DISTANCE currently
	std::vector<int> farthestPointSampling(int num_samples);
	Eigen::MatrixXd getRadialBaseFunctions(std::vector<int>& sampled_verts, double r, bool partition_of_one = true);
	Eigen::MatrixXd createSkinningSpace(int num_samples);

};

#endif
