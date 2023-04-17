#include "evaluationmetric.h"

#include <iostream>
#include <fstream>

#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
//#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/hausdorff.h>

#include "util.h"

EvaluationMetric::EvaluationMetric(const TriangleMesh* result, const TriangleMesh* target) : result(result), target(target) {
	Eigen::MatrixXd v1 = result->getVertices();
	Eigen::MatrixXi t1 = result->getTriangles();

	Eigen::MatrixXd v2 = target->getVertices();
	Eigen::MatrixXi t2 = target->getTriangles();

	this->result_volume = result->getVolume();
	this->target_volume = target->getVolume();

	// SKIPPING FOR NOW
	//// Intersection
	//this->intersection_volume = this->computeVolumeWithBooleanOp(v1, t1, v2, t2, "i", "intersection");
	//std::cout << "Intersection volume: " << this->intersection_volume << std::endl;

	////// Union
	////this->union_volume = this->computeVolumeWithBooleanOp(v1, t1, v2, t2, "u", "union");
	////std::cout << "Union volume: " << this->union_volume << std::endl;

	//// Diff A - B (t1 - t2)
	//this->diff_A_B_volume = this->computeVolumeWithBooleanOp(v1, t1, v2, t2, "m", "diff_t1_t2_volume");
	//std::cout << "diff result - target volume: " << this->diff_A_B_volume << std::endl;

	//// Diff B - A (t2 - t1)
	//this->diff_B_A_volume = this->computeVolumeWithBooleanOp(v2, t2, v1, t1, "m", "diff_t2_t1_volume");
	//std::cout << "diff target - result volume: " << this->diff_B_A_volume << std::endl;
}

// Compute volume after the boolean operation on mesh 1 and mesh 2
//double EvaluationMetric::computeVolumeWithBooleanOp(const Eigen::MatrixXd& v1, const Eigen::MatrixXi& t1, const Eigen::MatrixXd& v2, const Eigen::MatrixXi& t2, std::string boolean_operation, std::string file_name) {
//	M = { {v2, t2}, {v1, t1}, boolean_operation };
//	Eigen::MatrixXd bool_verts = M.cast_V<Eigen::MatrixXd>();
//	Eigen::MatrixXi bool_tri = M.F();
//
//	// Compute the volume
//	double bool_volume = computeTriangleMeshVolume(bool_verts, bool_tri);
//	return bool_volume;
//}

double EvaluationMetric::computeDiceCoefficient() {
	return (2 * this->intersection_volume) / (2 * this->diff_A_B_volume + this->diff_A_B_volume + this->diff_B_A_volume);
}


// squared euclidean distance of centroids of implicit meshes IN METERS 
double EvaluationMetric::computeCentroidDistance() {
	Eigen::Vector3d c1 = result->getCenter();
	Eigen::Vector3d c2 = target->getCenter();

	return (c2 - c1).norm();
}

double EvaluationMetric::computeHausdorffDist() {
	double dist;
	igl::hausdorff(result->getVertices(), result->getTriangles(), target->getVertices(), target->getTriangles(), dist);
	return dist;
}


void EvaluationMetric::writeScoresToFile(std::string folder, std::string file_name) {
	std::ofstream file;
	file.open(folder + file_name);
	file << "implicit_mesh_dice,centroid_euclidean_dist,implicit_mesh_hausdorff_dist" << std::endl;
	file << this->computeDiceCoefficient() << "," << this->computeCentroidDistance() << "," << this->computeHausdorffDist();
	file.close();
}
