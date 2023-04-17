#ifndef EVALUATION_METRIC_H
#define EVALUATION_METRIC_H

#include <string>

#include <Eigen/Dense>
//#include <igl/copyleft/cgal/CSGTree.h>

#include "tetrahedralmesh.h"


class EvaluationMetric {
private:
	//target = ground truth
	const TriangleMesh* result;
	const TriangleMesh* target;

	double result_volume;
	double target_volume;

	double intersection_volume;
	double union_volume;
	double diff_A_B_volume; // A = result B = target
	double diff_B_A_volume;

	// Store the CSG tree
	//igl::copyleft::cgal::CSGTree M;

public:
	EvaluationMetric(const TriangleMesh* result, const TriangleMesh* target);

	// Compute volume after the boolean operation on mesh 1 and mesh 2
	double computeVolumeWithBooleanOp(const Eigen::MatrixXd& v1, const Eigen::MatrixXi& t1, const Eigen::MatrixXd& v2, const Eigen::MatrixXi& t2, std::string boolean_operation, std::string file_name);

	double computeDiceCoefficient();
	// squared euclidean distance of centroids of implicit meshes IN METERS 
	double computeCentroidDistance();
	double computeHausdorffDist();

	void writeScoresToFile(std::string folder, std::string file_name);
};

#endif
