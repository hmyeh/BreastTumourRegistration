#include "deformationgradient.h"
#include "util.h"
#include <iostream>

DeformationGradient::DeformationGradient(const Eigen::Matrix3d& F) : F(F) {
	// SVD rotation variant from dynamic deformables
	SVD_RV(F, this->U, this->Sigma, this->V);

	// Use SVD to get Polar Decomposition 
	this->R = this->U * this->V.transpose();
	this->S = this->V * this->Sigma.asDiagonal() * this->V.transpose();
}


// Getter implementations for deformation gradient
Eigen::Matrix3d DeformationGradient::getF() const {
	return this->F;
}

Eigen::Matrix3d DeformationGradient::getU() const {
	return this->U;
}

Eigen::Vector3d DeformationGradient::getSigma() const {
	return this->Sigma;
}

Eigen::Matrix3d DeformationGradient::getV() const {
	return this->V;
}

// getters for stored Polar Decomposition rotation and stretching matrix
Eigen::Matrix3d DeformationGradient::getR() const {
	return this->R;
}

Eigen::Matrix3d DeformationGradient::getS() const {
	return this->S;
}
 
