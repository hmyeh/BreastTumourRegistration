#ifndef ELEMENT_H
#define ELEMENT_H

#include "materialmodel.h"
#include "util.h"
#include "deformationgradient.h"
#include <vector>
#include <iostream>



/*
Abstract class for a Finite Element
*/
class Element {
public:
	// Class for holding the USER defined parameters for an element
	struct Parameters {
		// Need pointer due to object slicing otherwise for material model
		std::unique_ptr<MaterialModel> material_model;
		// The density of the element (for now state it per element)
		double density;

		Parameters(std::unique_ptr<MaterialModel> material_model, double density);
	};

protected:
	// Dm^-1 rest pose stored
	Eigen::MatrixXd inverseDm;
	// Bm for forces computation (doing it as in Invertible Finite Elements for Large Deformation (this is also used in SNH reference code))
	Eigen::MatrixXd Bm;
	// The weight/volume/area/... of the element to see its contribution to the entire model
	double weight; 
	// Stores the indices to the vertices of the element
	std::vector<int> indices; 
	// store rest positions
	std::vector<Eigen::VectorXd> x;

	/// Current reason for using a pointer is to be able to update the parameters later on if something needs to be changed per area/element.
	Element::Parameters* param;

public:
	// Element constructor receiving parameters
	Element(std::vector<int> indices, Element::Parameters* param);

	// TODO: how to deal with deformation gradient function vs the set types for 3d vertices...
	virtual DeformationGradient computeDeformationGradient(std::vector<Eigen::VectorXd> x) = 0;//Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, 1, true> > x) = 0;
	virtual Eigen::MatrixXd computePFPx() = 0;
	virtual Eigen::MatrixXd computeBM(std::vector<Eigen::VectorXd> x) = 0;

	// Already implemented functions
	// Compute the strain energy using the material model and the deformation gradient F
	double computeStrainEnergy(const DeformationGradient& defGrad);
	// Compute the strain energy gradient using the material model and the deformation gradient F
	Eigen::VectorXd computeStrainGradient(const DeformationGradient& defGrad);
	// Compute the strain energy Hessian using the material model and the deformation gradient F
	Eigen::MatrixXd computeStrainHessian(const DeformationGradient& defGrad);
	// Getter for the weight
	double getWeight();
	// Getter for the density
	double getDensity(); // SHould this be exposed like this or should the element::parameters be exposed instead??
	// Getter for the indices
	std::vector<int> getIndices();
	// Getter for Bm
	Eigen::MatrixXd getBm();
	// For now putting a getter for the element parameters
	Element::Parameters* getParams();
};


// Always 4 vertices tet
class Tetrahedron : public Element{
public:
	// Constructor for a tetrahedron with references to each vertex row x, indices for the tetrahedron and a pointer to the material model
	// need to put column vectors in this
	Tetrahedron(std::vector<Eigen::VectorXd> x, std::vector<int> indices, Element::Parameters* param);

	// Compute the deformation gradient given the deformed vertex positions x for the correct indices
	// put column vectors in here
	DeformationGradient computeDeformationGradient(std::vector<Eigen::VectorXd> x);
	// Compute the PF/Px matrix
	Eigen::MatrixXd computePFPx();

	Eigen::MatrixXd computeBM(std::vector<Eigen::VectorXd> x);
};

#endif
