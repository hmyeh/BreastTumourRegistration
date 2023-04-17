#ifndef INVARIANT_H
#define INVARIANT_H

#include <Eigen/Dense>
#include "deformationgradient.h"



// Smith et al (2019) invariants
// dynamic deformables page 69 pdf 
// NOTE THAT THESE INVARIANTS ARE IMPLEMENTED ONLY FOR 3D...
class Invariant1 {
public:
	static double computeValue(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V);
	static Eigen::VectorXd computeGradient(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V);
	static Eigen::MatrixXd computeHessian(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V);
};


class Invariant2 {
public:
	static double computeValue(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V);
	static Eigen::VectorXd computeGradient(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V);
	static Eigen::MatrixXd computeHessian(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V);
};


class Invariant3 {
public:
	static double computeValue(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V);
	static Eigen::VectorXd computeGradient(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V);
	static Eigen::MatrixXd computeHessian(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V);
};

#endif
