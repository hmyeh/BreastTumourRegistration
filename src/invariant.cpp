#include "invariant.h"
#include "util.h"

double Invariant1::computeValue(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V) {
	// Compute Polar decomposition S matrix from SVD
	Eigen::Matrix3d S = V * Sigma.asDiagonal() * V.transpose();
	return S.trace();
}

Eigen::VectorXd Invariant1::computeGradient(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V) {
	// Compute Polar decomposition R (= U * V^T) matrix from SVD
	Eigen::Matrix3d gradient = V * V.transpose();
	Eigen::Map<Eigen::VectorXd> vectorizedGrad(gradient.data(), gradient.size());
	return vectorizedGrad;
}

Eigen::MatrixXd Invariant1::computeHessian(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V) {
	// Setup eigenmatrices
	double normalization = 1.0 / sqrt(2);
	Eigen::MatrixXd Q0(3, 3);
	Q0 << 0, -1, 0,
		1, 0, 0,
		0, 0, 0;
	Q0 = normalization * U * Q0 * V.transpose();
	Eigen::Map<Eigen::VectorXd> vecQ0(Q0.data(), Q0.size());

	Eigen::MatrixXd Q1(3, 3);
	Q1 << 0, 0, 0,
		0, 0, 1,
		0, -1, 0;
	Q1 = normalization * U * Q1 * V.transpose();
	Eigen::Map<Eigen::VectorXd> vecQ1(Q1.data(), Q1.size());

	Eigen::MatrixXd Q2(3, 3);
	Q2 << 0, 0, 1,
		0, 0, 0,
		-1, 0, 0;
	Q2 = normalization * U * Q2 * V.transpose();
	Eigen::Map<Eigen::VectorXd> vecQ2(Q2.data(), Q2.size());

	// Setup eigenvalues
	double lambda0 = 2.0 / (Sigma(0), Sigma(1));
	double lambda1 = 2.0 / (Sigma(1), Sigma(2));
	double lambda2 = 2.0 / (Sigma(0), Sigma(2));

	Eigen::MatrixXd vecHessian = lambda0 * vecQ0 * vecQ0.transpose() + lambda1 * vecQ1 * vecQ1.transpose() + lambda2 * vecQ2 * vecQ2.transpose();
	return vecHessian;
}



// Invariant2 implementations

double Invariant2::computeValue(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V) {
	//Eigen::Matrix3d F = deformation_gradient.getF();
	return (F.transpose() * F).trace();
}

Eigen::VectorXd Invariant2::computeGradient(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V) {
	//Eigen::Matrix3d F = deformation_gradient.getF();
	Eigen::MatrixXd gradient = 2 * F;
	Eigen::Map<Eigen::VectorXd> vectorizedGrad(gradient.data(), gradient.size());
	return vectorizedGrad;
}

Eigen::MatrixXd Invariant2::computeHessian(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V) {
	Eigen::MatrixXd hessian = 2 * Eigen::MatrixXd::Identity(9, 9);
	return hessian;
}


// Invariant 3 implementations
double Invariant3::computeValue(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V) {
	return F.determinant();
}

Eigen::VectorXd Invariant3::computeGradient(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V) {
	Eigen::MatrixXd gradient(3, 3);
	gradient << F.col(1).cross(F.col(2)), F.col(2).cross(F.col(0)), F.col(0).cross(F.col(1));
	Eigen::Map<Eigen::VectorXd> vectorizedGrad(gradient.data(), gradient.size());
	return vectorizedGrad;
}

Eigen::MatrixXd Invariant3::computeHessian(const Eigen::Matrix3d& F, const Eigen::Matrix3d& U, const Eigen::Vector3d& Sigma, const Eigen::Matrix3d& V) {
	Eigen::MatrixXd hessian(9, 9);
	Eigen::Matrix3d fHat0 = getCrossProductMatrix(F.col(0));
	Eigen::Matrix3d fHat1 = getCrossProductMatrix(F.col(1));
	Eigen::Matrix3d fHat2 = getCrossProductMatrix(F.col(2));
	hessian << Eigen::Matrix3d::Zero(), -fHat2, fHat1,
		fHat2, Eigen::Matrix3d::Zero(), -fHat0,
		-fHat1, fHat0, Eigen::Matrix3d::Zero();
	return hessian;
}
