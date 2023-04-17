#ifndef ENERGY_H
#define ENERGY_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

#include "element.h"
#include "tetrahedralmesh.h"
#include "constraint.h"

class Scene;

struct ElasticEnergy {
	static double computeEnergy(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x);
	static Eigen::VectorXd computeGradient(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x);
	static Eigen::SparseMatrix<double> computeHessian(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x);
};

struct GravitationalEnergy {
	static double computeEnergy(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x, const Eigen::Vector3d& g);
	static Eigen::VectorXd computeGradient(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x, const Eigen::Vector3d& g);
	static Eigen::SparseMatrix<double> computeHessian(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x, const Eigen::Vector3d& g);
};

struct PenaltyEnergy {
	static double computeEnergy(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x, const std::vector<std::shared_ptr<SoftConstraint> >& constraints, double weight);
	static Eigen::VectorXd computeGradient(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x, const std::vector<std::shared_ptr<SoftConstraint> >& constraints, double weight);
	static Eigen::SparseMatrix<double> computeHessian(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x, const std::vector<std::shared_ptr<SoftConstraint> >& constraints, double weight);
};




class EnergyFunction {
private:
	// Explicit energies
	//ElasticEnergy elastic_energy;
	//GravitationalEnergy gravitational_energy;
	//PenaltyEnergy penalty_energy;

	// Tetrahedral mesh used for the energy function (elements)
	TetrahedralMesh* tet_mesh;

	// std::vector containing the pointers to all the elements
	//std::vector<Element*> elements;
	std::vector<std::unique_ptr<Element> > elements;
	// Constraintmanager to store all the soft and hard constraints
	ConstraintManager constraint_manager;

	// Material parameters for each area
	std::vector<std::unique_ptr<Element::Parameters> > area_params;
	
	double penalty_weight;

	// Bool for gravity on or off
	bool gravity_on;
	Eigen::Vector3d gravity;
	// Number of vertices in tet mesh
	int num_verts;
public:
	EnergyFunction() {}
	EnergyFunction(TetrahedralMesh* tet_mesh, std::vector<std::unique_ptr<Element::Parameters> > area_params, double penalty_weight, bool gravity_on, Eigen::Vector3d gravity);

	// Create the finite elements (tetrahedrons) and be able to assign different element parameters to each area of the tet mesh
	void createElements();

	// Functions to get the energy, gradient and hessian respectively
	double computeEnergy(const Eigen::Matrix3Xd& x);
	Eigen::VectorXd computeGradient(const Eigen::Matrix3Xd& x, bool fixedDOFs = false);
	Eigen::SparseMatrix<double> computeHessian(const Eigen::Matrix3Xd& x, bool fixedDOFs = false);

	double computeElasticEnergy(const Eigen::Matrix3Xd& x);

	// Separate function for getting mass matrix
	Eigen::SparseMatrix<double> getMassMatrix();


	// Getter functions
	int getNumFixedVerts();

	ConstraintManager* getConstraintManager();
	const std::vector<std::shared_ptr<HardConstraint> >& getHardConstraints();
	const std::vector<std::unique_ptr<Element::Parameters> >& getMaterialParams();

	TetrahedralMesh* getTetMesh();

	// Setter functions
	void setGravityOn(bool gravity_on);
	void setGravityVector(Eigen::Vector3d gravity);
	// Getter functions
	Eigen::Vector3d getGravity();
	bool getGravityOn() { return this->gravity_on; }
};


#endif
