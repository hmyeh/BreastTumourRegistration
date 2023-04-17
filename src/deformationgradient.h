#ifndef DEFORMATIONGRADIENT_H
#define DEFORMATIONGRADIENT_H

#include <Eigen/Dense>

class DeformationGradient {
private:
	Eigen::Matrix3d F;

	// SVD (rotation variant)
	Eigen::Matrix3d U;
	Eigen::Vector3d Sigma;
	Eigen::Matrix3d V;

	// Polar Decomposition (rotation and stretching matrix)
	Eigen::Matrix3d R;
	Eigen::Matrix3d S;
public:
	DeformationGradient() {}
	DeformationGradient(const Eigen::Matrix3d& F);

	// Getter for F
	Eigen::Matrix3d getF() const;

	// getters for stored Singular Value decomposition
	Eigen::Matrix3d getU() const;
	Eigen::Vector3d getSigma() const;
	Eigen::Matrix3d getV() const;

	// getters for stored Polar Decomposition rotation and stretching matrix
	Eigen::Matrix3d getR() const;
	Eigen::Matrix3d getS() const;

};

#endif
