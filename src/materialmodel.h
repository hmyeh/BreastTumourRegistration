#ifndef MATERIALMODEL_H
#define MATERIALMODEL_H

#include <Eigen/Dense>
#include <vector>
#include "invariant.h"
#include "deformationgradient.h"

// autodiff include
// The autodiff will be used for the isotropic hyperelastic material model class and any subclasses
#include <autodiff/forward/dual.hpp>

/*
Abstract class/interface for all material models to follow
*/
class MaterialModel {
public:
	MaterialModel();
	// The functions each material model needs to implement
	virtual double computeStrainEnergy(const DeformationGradient& defGrad) = 0;
	virtual Eigen::VectorXd computeGradient(const DeformationGradient& defGrad) = 0;
	virtual Eigen::MatrixXd computeHessian(const DeformationGradient& defGrad) = 0;
};


// Easiest material model to derive (given in dynamic deformables course)
class DirichletMaterial : public MaterialModel {
public:
	DirichletMaterial();
	// Compute the strain energy for the material model given the deformation gradient F
	double computeStrainEnergy(const DeformationGradient& defGrad);
	// Compute the strain energy gradient for the material model given the deformation gradient F
	// Currently returning the vectorized result: vec(dPsi/dF)
	Eigen::VectorXd computeGradient(const DeformationGradient& defGrad);
	// Compute the strain energy for the material model given the deformation gradient F
	Eigen::MatrixXd computeHessian(const DeformationGradient& defGrad);
};


// The member variables /this class expects only the Smith et al 2019 invariants
class IsotropicHyperelasticMaterialModel : public MaterialModel {

public:
	IsotropicHyperelasticMaterialModel(); // maybe there should be default values not entirely sure...

	virtual autodiff::dual2nd psi(autodiff::dual2nd I1, autodiff::dual2nd I2, autodiff::dual2nd I3) = 0;
	// Compute the strain energy for the material model given the deformation gradient F
	// not implementing this since it needs to be abstract, as well as it being specific to the material model
	double computeStrainEnergy(const DeformationGradient& defGrad); // implementing it using autodiff
	// Compute the strain energy gradient for the material model given the deformation gradient F
	// Currently returning the vectorized result: vec(dPsi/dF)
	Eigen::VectorXd computeGradient(const DeformationGradient& defGrad);
	// Compute the strain energy for the material model given the deformation gradient F
	Eigen::MatrixXd computeHessian(const DeformationGradient& defGrad);
};


// for now use the Bonet and Wood 2008 Neo-hookean
// see page 69-70 in dynamic deformables pdf
class NeoHookeanMaterialModel : public IsotropicHyperelasticMaterialModel {
private:
	double mu;
	double lambda;
public:
	// E = Young's modulus
	// v = Poisson's ratio
	// These variables are used to compute the Lame parameters lambda and mu  (see page 77 dynamic deformables pdf)
	NeoHookeanMaterialModel(double mu, double lambda);
	autodiff::dual2nd psi(autodiff::dual2nd I1, autodiff::dual2nd I2, autodiff::dual2nd I3);
	//double computeStrainEnergy(const Eigen::Ref<const Eigen::MatrixXd> F, const std::vector<std::unique_ptr<Invariant>>& invariants);
};

class ARAPMaterialModel : public IsotropicHyperelasticMaterialModel {
public:
	// E = Young's modulus
	// v = Poisson's ratio
	// These variables are used to compute the Lame parameters lambda and mu  (see page 77 dynamic deformables pdf)
	ARAPMaterialModel();
	autodiff::dual2nd psi(autodiff::dual2nd I1, autodiff::dual2nd I2, autodiff::dual2nd I3);
};


class StVenantKirchhoffMaterialModel : public IsotropicHyperelasticMaterialModel {
private:
	double mu;
	double lambda;
public:
	// E = Young's modulus
	// v = Poisson's ratio
	// These variables are used to compute the Lame parameters lambda and mu  (see page 77 dynamic deformables pdf)
	StVenantKirchhoffMaterialModel(double mu, double lambda);
	autodiff::dual2nd psi(autodiff::dual2nd I1, autodiff::dual2nd I2, autodiff::dual2nd I3);
};


class StableNeoHookeanModel : public IsotropicHyperelasticMaterialModel {
private:
	double mu;
	double lambda;
public:
	// E = Young's modulus
	// v = Poisson's ratio
	// These variables are used to compute the Lame parameters lambda and mu  (see page 77 dynamic deformables pdf)
	StableNeoHookeanModel(double mu, double lambda);
	autodiff::dual2nd psi(autodiff::dual2nd I1, autodiff::dual2nd I2, autodiff::dual2nd I3);
};

#endif
