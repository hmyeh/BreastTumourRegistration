#include "materialmodel.h"
#include <math.h>
#include <iostream>
#include <typeinfo>
#include "util.h"

// Maybe this should be an abstract constructor
MaterialModel::MaterialModel() {}

// Example dirichlet material computed with deformation gradient F
DirichletMaterial::DirichletMaterial() {}

double DirichletMaterial::computeStrainEnergy(const DeformationGradient& defGrad) {
	Eigen::Matrix3d F = defGrad.getF();
	return F.squaredNorm();
}

Eigen::VectorXd DirichletMaterial::computeGradient(const DeformationGradient& defGrad) {
	Eigen::Matrix3d F = defGrad.getF();
	Eigen::MatrixXd gradient = 2 * F;
	Eigen::Map<Eigen::VectorXd> vectorizedGrad(gradient.data(), gradient.size());
	return vectorizedGrad;
}

Eigen::MatrixXd DirichletMaterial::computeHessian(const DeformationGradient& defGrad) {
	Eigen::Matrix3d F = defGrad.getF();
	return 2 * Eigen::MatrixXd::Identity(9, 9);
}


// Isotropic hyperelastic material this will be an abstract class for which others can 
// Convert the youngs modulus and poissons ratio to the lame parameters

IsotropicHyperelasticMaterialModel::IsotropicHyperelasticMaterialModel()  {}

double IsotropicHyperelasticMaterialModel::computeStrainEnergy(const DeformationGradient& defGrad) {
	Eigen::Matrix3d F = defGrad.getF();
	Eigen::Matrix3d U = defGrad.getU();
	Eigen::Vector3d Sigma = defGrad.getSigma();
	Eigen::Matrix3d V = defGrad.getV();

	autodiff::dual2nd I1 = Invariant1::computeValue(F, U, Sigma, V);
	autodiff::dual2nd I2 = Invariant2::computeValue(F, U, Sigma, V);
	autodiff::dual2nd I3 = Invariant3::computeValue(F, U, Sigma, V);

	return val(this->psi(I1, I2, I3));
}

Eigen::VectorXd IsotropicHyperelasticMaterialModel::computeGradient(const DeformationGradient& defGrad) {
	Eigen::Matrix3d F = defGrad.getF();
	Eigen::Matrix3d U = defGrad.getU();
	Eigen::Vector3d Sigma = defGrad.getSigma();
	Eigen::Matrix3d V = defGrad.getV();

	autodiff::dual2nd I1 = Invariant1::computeValue(F, U, Sigma, V);
	autodiff::dual2nd I2 = Invariant2::computeValue(F, U, Sigma, V);
	autodiff::dual2nd I3 = Invariant3::computeValue(F, U, Sigma, V);


	//  figure 7.5 (pg102), 7.4 (pg98 with equations 7.14 7.15 pg 97), 7.6 (pg103), 7.7 (pg104)(last one is full algorithm overall stuff)
	//auto [PsiI1, PsiI2, PsiI3] = autodiff::derivatives(this->psi(I1, I2, I3), autodiff::wrt(I1, I2, I3));
	double PsiI1 = autodiff::derivative([&](auto I1, auto I2, auto I3) { return psi(I1, I2, I3); }, autodiff::wrt(I1), autodiff::at(I1, I2, I3));
	double PsiI2 = autodiff::derivative([&](auto I1, auto I2, auto I3) { return psi(I1, I2, I3); }, autodiff::wrt(I2), autodiff::at(I1, I2, I3));
	double PsiI3 = autodiff::derivative([&](auto I1, auto I2, auto I3) { return psi(I1, I2, I3); }, autodiff::wrt(I3), autodiff::at(I1, I2, I3));

	Eigen::VectorXd vectorizedGrad = PsiI1 * Invariant1::computeGradient(F, U, Sigma, V) + PsiI2 * Invariant2::computeGradient(F, U, Sigma, V) + PsiI3 * Invariant3::computeGradient(F, U, Sigma, V);

	return vectorizedGrad;
}

Eigen::MatrixXd IsotropicHyperelasticMaterialModel::computeHessian(const DeformationGradient& defGrad) {
	// Get F and its svd (rotation variant)
	Eigen::Matrix3d F = defGrad.getF();
	Eigen::Vector3d sigma = defGrad.getSigma();
	Eigen::Matrix3d U = defGrad.getU();
	Eigen::Matrix3d V = defGrad.getV();

	// Compute the gradients for the invariants
	Eigen::VectorXd gradI1 = Invariant1::computeGradient(F, U, sigma, V);
	Eigen::VectorXd gradI2 = Invariant2::computeGradient(F, U, sigma, V);
	Eigen::VectorXd gradI3 = Invariant3::computeGradient(F, U, sigma, V);

	// Compute the hessians for the invariants
	Eigen::MatrixXd hesI1 = Invariant1::computeHessian(F, U, sigma, V);
	Eigen::MatrixXd hesI2 = Invariant2::computeHessian(F, U, sigma, V);
	Eigen::MatrixXd hesI3 = Invariant3::computeHessian(F, U, sigma, V);

	// Invariant values
	//  figure 7.5 (pg102), 7.4 (pg98 with equations 7.14 7.15 pg 97), 7.6 (pg103), 7.7 (pg104)(last one is full algorithm overall stuff)
	autodiff::dual2nd I1Var = Invariant1::computeValue(F, U, sigma, V);
	autodiff::dual2nd I2Var = Invariant2::computeValue(F, U, sigma, V);
	autodiff::dual2nd I3Var = Invariant3::computeValue(F, U, sigma, V);

	double I1 = val(I1Var);
	double I2 = val(I2Var);
	double I3 = val(I3Var);

	//  figure 7.5 (pg102), 7.4 (pg98 with equations 7.14 7.15 pg 97), 7.6 (pg103), 7.7 (pg104)(last one is full algorithm overall stuff)
	auto [_psi1, PsiI1, PsiI1I2] = autodiff::derivatives([&](auto I1, auto I2, auto I3) { return psi(I1, I2, I3); }, autodiff::wrt(I1Var, I2Var), autodiff::at(I1Var, I2Var, I3Var));
	auto [_psi2, PsiI2, PsiI2I3] = autodiff::derivatives([&](auto I1, auto I2, auto I3) { return psi(I1, I2, I3); }, autodiff::wrt(I2Var, I3Var), autodiff::at(I1Var, I2Var, I3Var));
	auto [_psi3, PsiI3, PsiI3I1] = autodiff::derivatives([&](auto I1, auto I2, auto I3) { return psi(I1, I2, I3); }, autodiff::wrt(I3Var, I1Var), autodiff::at(I1Var, I2Var, I3Var));

	auto PsiI1I1 = autodiff::derivatives([&](auto I1, auto I2, auto I3) { return psi(I1, I2, I3); }, autodiff::wrt(I1Var, I1Var), autodiff::at(I1Var, I2Var, I3Var))[2];
	auto PsiI2I2 = autodiff::derivatives([&](auto I1, auto I2, auto I3) { return psi(I1, I2, I3); }, autodiff::wrt(I2Var, I2Var), autodiff::at(I1Var, I2Var, I3Var))[2];
	auto PsiI3I3 = autodiff::derivatives([&](auto I1, auto I2, auto I3) { return psi(I1, I2, I3); }, autodiff::wrt(I3Var, I3Var), autodiff::at(I1Var, I2Var, I3Var))[2];


	// Compute eigenvalues counting as in dynamic deformables paper pg97
	Eigen::VectorXd lambda(9);
	// z- twist
	lambda(3) = (2.0 / (sigma(0) + sigma(1))) * PsiI1 + 2 * PsiI2 + sigma(2) * PsiI3;
	//x- twist
	lambda(4) = (2.0 / (sigma(1) + sigma(2))) * PsiI1 + 2 * PsiI2 + sigma(0) * PsiI3;
	// y- twist
	lambda(5) = (2.0 / (sigma(0) + sigma(2))) * PsiI1 + 2 * PsiI2 + sigma(1) * PsiI3;

	//x-flip
	lambda(6) = 2 * PsiI2 - sigma(2) * PsiI3;
	//y-flip
	lambda(7) = 2 * PsiI2 - sigma(0) * PsiI3;
	//z-flip
	lambda(8) = 2 * PsiI2 - sigma(1) * PsiI3;

	// If you win the eigenvalue jackpot: these are the analytic eigenvalues for scaling
	// x-scale
	lambda(0) = 2 * PsiI2 + PsiI1I1 + 4 * (sigma(0) * sigma(0)) * PsiI2I2 + (sigma(1) * sigma(1)) * (sigma(2) * sigma(2)) * PsiI3I3 + 4 * sigma(0) * PsiI1I2 + 4 * I3 * PsiI2I3 + 2 * sigma(1) * sigma(2) * PsiI3I1;
	// y-scale
	lambda(1) = 2 * PsiI2 + PsiI1I1 + 4 * (sigma(1) * sigma(1)) * PsiI2I2 + (sigma(0) * sigma(0)) * (sigma(2) * sigma(2)) * PsiI3I3 + 4 * sigma(1) * PsiI1I2 + 4 * I3 * PsiI2I3 + 2 * sigma(0) * sigma(2) * PsiI3I1;
	// z-scale
	lambda(2) = 2 * PsiI2 + PsiI1I1 + 4 * (sigma(2) * sigma(2)) * PsiI2I2 + (sigma(0) * sigma(0)) * (sigma(1) * sigma(1)) * PsiI3I3 + 4 * sigma(2) * PsiI1I2 + 4 * I3 * PsiI2I3 + 2 * sigma(0) * sigma(1) * PsiI3I1;


	// setup corresponding eigenvectors and sum everything  (see Analytic Eigensystems for Isotropic Distortion Energies algorithm 1 pg9)
	double inverseSqrtTwo = 1.0 / std::sqrt(2.0);

	// twist matrices
	Eigen::MatrixXd Q3(3, 3);
	Q3 << 0, -1, 0,
		1, 0, 0,
		0, 0, 0;
	Q3 = inverseSqrtTwo * U * Q3 * V.transpose();
	Eigen::Map<Eigen::VectorXd> vecQ3(Q3.data(), Q3.size());

	Eigen::MatrixXd Q4(3, 3);
	Q4 << 0, 0, 0,
		0, 0, 1,
		0, -1, 0;
	Q4 = inverseSqrtTwo * U * Q4 * V.transpose();
	Eigen::Map<Eigen::VectorXd> vecQ4(Q4.data(), Q4.size());

	Eigen::MatrixXd Q5(3, 3);
	Q5 << 0, 0, 1,
		0, 0, 0,
		-1, 0, 0;
	Q5 = inverseSqrtTwo * U * Q5 * V.transpose();
	Eigen::Map<Eigen::VectorXd> vecQ5(Q5.data(), Q5.size());

	// flip matrices
	Eigen::MatrixXd Q6(3, 3);
	Q6 << 0, 1, 0,
		1, 0, 0,
		0, 0, 0;
	Q6 = inverseSqrtTwo * U * Q6 * V.transpose();
	Eigen::Map<Eigen::VectorXd> vecQ6(Q6.data(), Q6.size());

	Eigen::MatrixXd Q7(3, 3);
	Q7 << 0, 0, 0,
		0, 0, 1,
		0, 1, 0;
	Q7 = inverseSqrtTwo * U * Q7 * V.transpose();
	Eigen::Map<Eigen::VectorXd> vecQ7(Q7.data(), Q7.size());

	Eigen::MatrixXd Q8(3, 3);
	Q8 << 0, 0, 1,
		0, 0, 0,
		1, 0, 0;
	Q8 = inverseSqrtTwo * U * Q8 * V.transpose();
	Eigen::Map<Eigen::VectorXd> vecQ8(Q8.data(), Q8.size());

	// Setup for last 3 eigenvector matrices
	Eigen::MatrixXd D0(3, 3);
	D0 << 1, 0, 0,
		0, 0, 0,
		0, 0, 0;
	D0 = U * D0 * V.transpose();

	Eigen::MatrixXd D1(3, 3);
	D1 << 0, 0, 0,
		0, 1, 0,
		0, 0, 0;
	D1 = U * D1 * V.transpose();

	Eigen::MatrixXd D2(3, 3);
	D2 << 0, 0, 0,
		0, 0, 0,
		0, 0, 1;
	D2 = U * D2 * V.transpose();

	Eigen::MatrixXd Q0 = Eigen::MatrixXd(D0);
	Eigen::MatrixXd Q1 = Eigen::MatrixXd(D1);
	Eigen::MatrixXd Q2 = Eigen::MatrixXd(D2);

	// get stretching system A, and then check if eigen value jackpot has been hit figure 7.4 pg98/97 dynamic deformables
	Eigen::MatrixXd A(3, 3);
	A.setZero();

	A(0, 0) = 2.0 * PsiI2 + PsiI1I1 + 4.0 * sigma(0) * sigma(0) * PsiI2I2 + (I3 / sigma(0)) * (I3 / sigma(0)) * PsiI3I3 + 4.0 * sigma(0) * PsiI1I2 + 4.0 * I3 * PsiI2I3 + 2.0 * (I3 / sigma(0)) * PsiI3I1;
	A(1, 1) = 2.0 * PsiI2 + PsiI1I1 + 4.0 * sigma(1) * sigma(1) * PsiI2I2 + (I3 / sigma(1)) * (I3 / sigma(1)) * PsiI3I3 + 4.0 * sigma(1) * PsiI1I2 + 4.0 * I3 * PsiI2I3 + 2.0 * (I3 / sigma(1)) * PsiI3I1;
	A(2, 2) = 2.0 * PsiI2 + PsiI1I1 + 4.0 * sigma(2) * sigma(2) * PsiI2I2 + (I3 / sigma(2)) * (I3 / sigma(2)) * PsiI3I3 + 4.0 * sigma(2) * PsiI1I2 + 4.0 * I3 * PsiI2I3 + 2.0 * (I3 / sigma(2)) * PsiI3I1;

	A(0, 1) = sigma(2) * PsiI3 + PsiI1I1 + 4.0 * (I3 / sigma(2)) * PsiI2I2 + sigma(2) * I3 * PsiI3I3 + 2.0 * sigma(2) * (I2 - (sigma(2), sigma(2))) * PsiI2I3 + (I1 - sigma(2)) * (sigma(2) * PsiI3I1 + 2.0 * PsiI1I2);
	A(0, 2) = sigma(1) * PsiI3 + PsiI1I1 + 4.0 * (I3 / sigma(1)) * PsiI2I2 + sigma(1) * I3 * PsiI3I3 + 2.0 * sigma(1) * (I2 - (sigma(1), sigma(1))) * PsiI2I3 + (I1 - sigma(1)) * (sigma(1) * PsiI3I1 + 2.0 * PsiI1I2);
	A(1, 2) = sigma(0) * PsiI3 + PsiI1I1 + 4.0 * (I3 / sigma(0)) * PsiI2I2 + sigma(0) * I3 * PsiI3I3 + 2.0 * sigma(0) * (I2 - (sigma(0), sigma(0))) * PsiI2I3 + (I1 - sigma(0)) * (sigma(0) * PsiI3I1 + 2.0 * PsiI1I2);

	A(1, 0) = A(0, 1);
	A(2, 0) = A(0, 2);
	A(2, 1) = A(1, 2);


	// Check eigenvalue jackpot or not
	double jackpot = A(0, 1) + A(0, 2) + A(1, 2);
	if (abs(jackpot) > 1e-4) {
		// when you lost the jackpot
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> stretchingSystemEigenDecomp(A);
		Eigen::VectorXd eigenvalues = stretchingSystemEigenDecomp.eigenvalues();
		Eigen::MatrixXd eigenvectors = stretchingSystemEigenDecomp.eigenvectors();


		// Set eigenvalues for hessian to eigenvalues of stretching system A
		lambda(0) = eigenvalues(0);
		lambda(1) = eigenvalues(1);
		lambda(2) = eigenvalues(2);

		// Following according to Analytic eigensystems for isotropic distortion energies paper page 8 pdf
		// See page 15 isotropic eigensystems F.3 st venant krichhoff, here it simply uses the eigenvectors 
		Q0 = U * eigenvectors.col(0).asDiagonal() * V.transpose();
		Q1 = U * eigenvectors.col(1).asDiagonal() * V.transpose();
		Q2 = U * eigenvectors.col(2).asDiagonal() * V.transpose();
	}

	// scaling eigen vector matrices
	Eigen::Map<Eigen::VectorXd> vecQ0(Q0.data(), Q0.size());
	Eigen::Map<Eigen::VectorXd> vecQ1(Q1.data(), Q1.size());
	Eigen::Map<Eigen::VectorXd> vecQ2(Q2.data(), Q2.size());

	// Store all eigenvectors into Q
	Eigen::MatrixXd Q(9, 9);
	Q.col(0) = vecQ0;
	Q.col(1) = vecQ1;
	Q.col(2) = vecQ2;
	Q.col(3) = vecQ3;
	Q.col(4) = vecQ4;
	Q.col(5) = vecQ5;
	Q.col(6) = vecQ6;
	Q.col(7) = vecQ7;
	Q.col(8) = vecQ8;

	// Clamp eigenvalues below 0.0
	for (unsigned int i = 0; i < lambda.size(); ++i) {
		lambda(i) = fmax(lambda(i), 1e-4); 
	}

	// Compute projected Hessian
	Eigen::MatrixXd projected_hessian = Q * lambda.asDiagonal() * Q.transpose();
	return projected_hessian;
}
// Autodiff isotropic hyperelastic material model stuff


// Neo hookean material model Bonet and wood 2008 implementation
// See page 69-70 dynamic deformables pdf
NeoHookeanMaterialModel::NeoHookeanMaterialModel(double mu, double lambda): mu(mu), lambda(lambda) {}

autodiff::dual2nd NeoHookeanMaterialModel::psi(autodiff::dual2nd I1, autodiff::dual2nd I2, autodiff::dual2nd I3) {
	return this->mu * 0.5 * (I2 - 3) - this->mu * log(I3) + this->lambda * 0.5 * (log(I3) * log(I3));
}


// ARAP material model

ARAPMaterialModel::ARAPMaterialModel() {}

autodiff::dual2nd ARAPMaterialModel::psi(autodiff::dual2nd I1, autodiff::dual2nd I2, autodiff::dual2nd I3) {
	return I2 - 2 * I1 + 3;
}

// See page 70 of Dynamic deformables

StVenantKirchhoffMaterialModel::StVenantKirchhoffMaterialModel(double mu, double lambda) : mu(mu), lambda(lambda) {
}

// See page 79 Dynamic deformables
autodiff::dual2nd StVenantKirchhoffMaterialModel::psi(autodiff::dual2nd I1, autodiff::dual2nd I2, autodiff::dual2nd I3) {
	autodiff::dual2nd IIc = 0.5 * (I2 * I2 - pow(I1, 4)) + I1 * I1 * I2 + 4 * I1 * I3;
	return (this->lambda / 8.0) * ((I2 - 3) * (I2 - 3)) + (this->mu / 8.0) * (8 * I1 * I3 + (I2 * I2) + 2 * (I1 * I1) * I2 - 4 * I2 - pow(I1, 4.0) + 6);
}

// Stable neohookean page 84 dynamic deformables
StableNeoHookeanModel::StableNeoHookeanModel(double mu, double lambda) : mu(mu), lambda(lambda) {
}

autodiff::dual2nd StableNeoHookeanModel::psi(autodiff::dual2nd I1, autodiff::dual2nd I2, autodiff::dual2nd I3) {
	return (this->mu * 0.5) * (I2 - 3) + (this->lambda * 0.5) * ((I3 - 1 - (this->mu / this->lambda)) * (I3 - 1 - (this->mu / this->lambda)));
}
