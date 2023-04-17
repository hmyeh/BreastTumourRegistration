#ifndef CONSTRAINT_H
#define CONSTRAINT_H

#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "tetrahedralmesh.h"


// Abstract base class Constraint
class Constraint {};
// Empty base class only to ensure all type of constraints inherit this class
// For now assuming that each constraint is linked only to a single vertex 
class PointConstraint : public Constraint {
private:
	int vertex_idx;
public:
	PointConstraint(int vertex_idx);
	int getVertexIdx();
};


// Hard constraints solved through Newton-KKT system

class HardConstraint : public PointConstraint {
public:
	HardConstraint(int vertex_idx);

	// constraint is Ax = b
	virtual Eigen::Vector3d getRHSb() = 0;
};

// Soft constraints solved through penalty method 
class SoftConstraint : public PointConstraint {
public:
	SoftConstraint(int vertex_idx);

	// For penalty method need to find solution through derivatives
	virtual double computeConstraint(const Eigen::Vector3d& x) = 0;
	virtual Eigen::Vector3d computeGradient(const Eigen::Vector3d& x) = 0;
	virtual Eigen::Matrix3d computeHessian(const Eigen::Vector3d& x) = 0; 
};


// Hard fixed point constraint
class HardFixedPointConstraint : public HardConstraint {
private:
	Eigen::Vector3d fixed_position;
public:
	HardFixedPointConstraint(int vertex_idx, Eigen::Vector3d fixed_position);
	Eigen::Vector3d getRHSb();
};

// Soft fixed point constraint
class SoftFixedPointConstraint : public SoftConstraint {
private:
	Eigen::Vector3d fixed_position;
	double weight;
public:
	SoftFixedPointConstraint(int vertex_idx, Eigen::Vector3d fixed_position, double weight);

	// For penalty method need to find solution through derivatives
	// the weight mu is left out since multiplied in the energy
	// x in this case is specific for the constraint (current position for constrained vertex)
	// constraint = c(x_k)
	double computeConstraint(const Eigen::Vector3d& x);
	// gradient = c(x_k) c'(x_k) 
	Eigen::Vector3d computeGradient(const Eigen::Vector3d& x);
	// hessian = c(x_k) c''(x_k) + c'(x_k)^T c'(x_k) 
	Eigen::Matrix3d computeHessian(const Eigen::Vector3d& x);
};


// Soft fixed point to tangent plane constraint (Assuming normal is unit normal// also safety normalized call)
class SoftPointToPlaneConstraint : public SoftConstraint {
private:
	Eigen::Vector3d plane_point;
	Eigen::Vector3d normal;
	double weight;
public:
	SoftPointToPlaneConstraint(int vertex_idx, Eigen::Vector3d plane_point, Eigen::Vector3d normal, double weight);

	// For penalty method need to find solution through derivatives
	// the weight mu is left out since multiplied in the energy
	// x in this case is specific for the constraint (current position for constrained vertex)
	// constraint = c(x_k)
	double computeConstraint(const Eigen::Vector3d& x);
	// gradient = c(x_k) c'(x_k) 
	Eigen::Vector3d computeGradient(const Eigen::Vector3d& x);
	// hessian = c(x_k) c''(x_k) + c'(x_k)^T c'(x_k) 
	Eigen::Matrix3d computeHessian(const Eigen::Vector3d& x);
};



// constraintmanager class for storing constraints
class ConstraintManager {
private:
	std::vector<std::shared_ptr<HardConstraint> > hard_constraints;
	std::vector<std::shared_ptr<SoftConstraint> > soft_constraints;
public:
	ConstraintManager() {}

	int getNumSoftConstraints();
	int getNumHardConstraints();

	void addHardConstraint(std::shared_ptr<HardConstraint> constraint);
	void addHardConstraints(std::vector<std::shared_ptr<HardConstraint> > constraints_list);
	void addSoftConstraint(std::shared_ptr<SoftConstraint> constraint);
	void addSoftConstraints(std::vector<std::shared_ptr<SoftConstraint> > constraints_list);

	const std::vector<std::shared_ptr<HardConstraint> >& getHardConstraints();
	const std::vector<std::shared_ptr<SoftConstraint> >& getSoftConstraints();

	// Needs to be called after energyfunction is created
	void addHardCorrespondenceConstraints(TetrahedralMesh* source, TetrahedralMesh* target, const Eigen::MatrixXi& correspondences, Eigen::MatrixXd& subspace);

	// Add correspondence constraints // Correspondence matrix: source_vertex, target_vertex
	void addCorrespondenceConstraints(TetrahedralMesh* source, TetrahedralMesh* target, const Eigen::MatrixXi& correspondences, double correspondence_weight = 1.0);

	void addFixedPointConstraints(TetrahedralMesh* tet_mesh, const std::vector<bool>& fixed_verts);
	// Overloading method for subspace version For the subspace it removes the Degrees of Freedom
	void addFixedPointConstraints(TetrahedralMesh* tet_mesh, const std::vector<bool>& fixed_verts, Eigen::MatrixXd& subspace);

	// Methods for clearing the constraints
	void clearSoftConstraints();
	void clearHardConstraints();

	// Soft fixed frame constraint (point with attached frame get fixed)
	static std::vector<std::shared_ptr<SoftConstraint>> createSoftFixedFrameConstraint(int initial_constr_vert_idx, int target_constr_vert_idx, int num_frame_rings,
		const Eigen::Matrix3Xd& initial_verts_pos, const Eigen::Matrix3Xd& initial_verts_normals, const Eigen::Matrix3Xd& target_verts_pos,
		const Eigen::Matrix3Xd& target_verts_normals, const Eigen::SparseMatrix<double>& adjacency_matrix);
};


#endif
