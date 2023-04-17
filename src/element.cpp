#include "element.h"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>


// ELEMENT FUNCTION IMPLEMENTATIONS

// Nested Parameters class constructor
Element::Parameters::Parameters(std::unique_ptr<MaterialModel> material_model, double density) : material_model(std::move(material_model)), density(density) {

}

// Initialize weight with 0.0
Element::Element(std::vector<int> indices, Element::Parameters* param): param(param), indices(indices), weight(0.0)  {}


double Element::computeStrainEnergy(const DeformationGradient& defGrad) {
	return this->param->material_model->computeStrainEnergy(defGrad);
}

Eigen::VectorXd Element::computeStrainGradient(const DeformationGradient& defGrad) {
	// dPsi/dx = vec(dF/dx)^T vec(dPsi/dF)
	Eigen::MatrixXd PFPx = this->computePFPx(); // this is vec(dF/dx)
	Eigen::VectorXd strainGrad = this->param->material_model->computeGradient(defGrad); // this is vec(dPsi/dF)
	return PFPx.transpose() * strainGrad;
	//return  strainGrad; 
}

Eigen::MatrixXd Element::computeStrainHessian(const DeformationGradient& defGrad) {
	// dPsi^2/dx^2 = vec(dF/dx)^T vec(dPsi^2/dF^2) vec(dF/dx)
	Eigen::MatrixXd PFPx = this->computePFPx(); // this is vec(dF/dx)
	Eigen::MatrixXd strainHes = this->param->material_model->computeHessian(defGrad); // this is vec(dPsi^2/dF^2)
	return PFPx.transpose() * strainHes * PFPx;
}

double Element::getWeight() {
	return this->weight;
}

double Element::getDensity() {
	return this->param->density;
}

std::vector<int> Element::getIndices() {
	return this->indices;
}

Eigen::MatrixXd Element::getBm() {
	return this->Bm;
}

Element::Parameters* Element::getParams() {
	return this->param;
}

// TETRAHEDRON FUNCTION IMPLEMENTATIONS

Tetrahedron::Tetrahedron(std::vector<Eigen::VectorXd> x, std::vector<int> indices, Element::Parameters* param) : Element(indices, param) {
	// Compute the inverse Dm matrix and store it
	Eigen::MatrixXd Dm(3, 3); // 3x3 Dm matrix
	Dm << x[1] - x[0], x[2] - x[0], x[3] - x[0];
	this->inverseDm = Dm.inverse();

	// TODO: prob perform this in parent class
	this->Bm = computeBM(x);
	// The weight is the volume of the tetrahedron = 1/6 * abs(det Dm)
	this->weight = 1.0 / 6.0 * abs(Dm.determinant());
}


DeformationGradient Tetrahedron::computeDeformationGradient(std::vector<Eigen::VectorXd> x) {
	// Checking if x has indeed 4 verts
	assert(x.size() == 4);
	
	// Compute the deformation gradient Ds Dm^-1
	Eigen::Matrix3d Ds; // 3x3 Ds matrix
	Ds << x[1] - x[0], x[2] - x[0], x[3] - x[0];

	Eigen::Matrix3d F = Ds * this->inverseDm;
	DeformationGradient defgrad(F);
	return defgrad;
}

// Specific matrix Bm used to compute forces in Stable Neohookean Reference code
// as well as in the Invertible Finite Elements for large deformation paper
Eigen::MatrixXd Tetrahedron::computeBM(std::vector<Eigen::VectorXd> x) {
	Eigen::Vector3d x0 = x[0];
	Eigen::Vector3d x1 = x[1];
	Eigen::Vector3d x2 = x[2];
	Eigen::Vector3d x3 = x[3];
	// TODO: make this cleaner
	Eigen::Vector3d tri0_normal = ((x1 - x0).cross(x3 - x0)).normalized();
	double tri0_area = 0.5 * ((x1 - x0).cross(x3 - x0)).norm();

	Eigen::Vector3d tri1_normal = ((x2 - x0).cross(x1 - x0)).normalized();
	double tri1_area = 0.5 * ((x2 - x0).cross(x1 - x0)).norm();

	Eigen::Vector3d tri2_normal = ((x2 - x3).cross(x0 - x3)).normalized();
	double tri2_area = 0.5 * ((x2 - x3).cross(x0 - x3)).norm();

	Eigen::Vector3d tri3_normal = ((x2 - x1).cross(x3 - x1)).normalized();
	double tri3_area = 0.5 * ((x2 - x1).cross(x3 - x1)).norm();

	Eigen::MatrixXd Bm(3, 4);
	// Calculate the area vectors
	   // v0 is incident on faces (0,1,2)
	Bm.col(0) = tri0_normal * tri0_area + tri1_normal * tri1_area + tri2_normal * tri2_area;
	// v1 is incident on faces (0,1,3)
	Bm.col(1) = tri0_normal * tri0_area + tri1_normal * tri1_area + tri3_normal * tri3_area;
	// v2 is incident on faces (1,2,3)
	Bm.col(2) = tri1_normal * tri1_area + tri2_normal * tri2_area + tri3_normal * tri3_area;
	// v3 is incident on faces (0,2,3)
	Bm.col(3) = tri0_normal * tri0_area + tri2_normal * tri2_area + tri3_normal * tri3_area;
	Bm /= -3.0;
	return Bm;
}

// For this function it is assumed Dminv is a 3x3 matrix
// Code from Dynamic deformables for dF/dx flattened version page 180
Eigen::MatrixXd Tetrahedron::computePFPx() {
	const double m = this->inverseDm(0, 0);
	const double n = this->inverseDm(0, 1);
	const double o = this->inverseDm(0, 2);
	const double p = this->inverseDm(1, 0);
	const double q = this->inverseDm(1, 1);
	const double r = this->inverseDm(1, 2);
	const double s = this->inverseDm(2, 0);
	const double t = this->inverseDm(2, 1);
	const double u = this->inverseDm(2, 2);

	const double t1 = -m - p - s;
	const double t2 = -n - q - t;
	const double t3 = -o - r - u;

	Eigen::MatrixXd PFPx(9, 12);
	PFPx.setZero();
	PFPx.setZero();
	PFPx(0, 0) = t1;
	PFPx(0, 3) = m;
	PFPx(0, 6) = p;
	PFPx(0, 9) = s;
	PFPx(1, 1) = t1;
	PFPx(1, 4) = m;
	PFPx(1, 7) = p;
	PFPx(1, 10) = s;
	PFPx(2, 2) = t1;
	PFPx(2, 5) = m;
	PFPx(2, 8) = p;
	PFPx(2, 11) = s;
	PFPx(3, 0) = t2;
	PFPx(3, 3) = n;
	PFPx(3, 6) = q;
	PFPx(3, 9) = t;
	PFPx(4, 1) = t2;
	PFPx(4, 4) = n;
	PFPx(4, 7) = q;
	PFPx(4, 10) = t;
	PFPx(5, 2) = t2;
	PFPx(5, 5) = n;
	PFPx(5, 8) = q;
	PFPx(5, 11) = t;
	PFPx(6, 0) = t3;
	PFPx(6, 3) = o;
	PFPx(6, 6) = r;
	PFPx(6, 9) = u;
	PFPx(7, 1) = t3;
	PFPx(7, 4) = o;
	PFPx(7, 7) = r;
	PFPx(7, 10) = u;
	PFPx(8, 2) = t3;
	PFPx(8, 5) = o;
	PFPx(8, 8) = r;
	PFPx(8, 11) = u;
	
	return PFPx;
}
