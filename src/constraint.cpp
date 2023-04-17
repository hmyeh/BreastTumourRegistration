#include "constraint.h"

#include <iostream>
#include <set>
#include <algorithm>
#include <math.h>

#include <igl/bfs.h>
#include <igl/rigid_alignment.h>


//// PointConstraint Implementations

PointConstraint::PointConstraint(int vertex_idx) : vertex_idx(vertex_idx) {}

int PointConstraint::getVertexIdx() { return this->vertex_idx; }

//// HardConstraint Implementations

HardConstraint::HardConstraint(int vertex_idx) : PointConstraint(vertex_idx) {}

//// SoftConstraint Implementations

SoftConstraint::SoftConstraint(int vertex_idx) : PointConstraint(vertex_idx) {}

//// HardFixedPointConstraint Implementations

HardFixedPointConstraint::HardFixedPointConstraint(int vertex_idx, Eigen::Vector3d fixed_position) : HardConstraint(vertex_idx), fixed_position(fixed_position) {}

Eigen::Vector3d HardFixedPointConstraint::getRHSb() {
	return this->fixed_position;
}

//// SoftFixedPointConstraint Implementations

SoftFixedPointConstraint::SoftFixedPointConstraint(int vertex_idx, Eigen::Vector3d fixed_position, double weight) : SoftConstraint(vertex_idx), fixed_position(fixed_position), weight(weight) {}

double SoftFixedPointConstraint::computeConstraint(const Eigen::Vector3d& x) {
	return weight * 0.5 * (x - fixed_position).squaredNorm();
}

// gradient = c(x_k) c'(x_k) 
Eigen::Vector3d SoftFixedPointConstraint::computeGradient(const Eigen::Vector3d& x) {
	return weight * (x - fixed_position);
}

// hessian = c(x_k) c''(x_k) + c'(x_k)^T c'(x_k) 
Eigen::Matrix3d SoftFixedPointConstraint::computeHessian(const Eigen::Vector3d& x) {
	return weight * Eigen::Matrix3d::Identity();
}

//// SoftPointToPlaneConstraint Implementations

SoftPointToPlaneConstraint::SoftPointToPlaneConstraint(int vertex_idx, Eigen::Vector3d plane_point, Eigen::Vector3d normal, double weight) : SoftConstraint(vertex_idx),
plane_point(plane_point), normal(normal.normalized()), weight(weight) {}

// For penalty method need to find solution through derivatives
// the weight mu is left out since multiplied in the energy
// x in this case is specific for the constraint (current position for constrained vertex)
// constraint = c(x_k)
double SoftPointToPlaneConstraint::computeConstraint(const Eigen::Vector3d& x) {
	double point_to_plane_dist = (x - this->plane_point).dot(this->normal);
	return weight * 0.5 * point_to_plane_dist * point_to_plane_dist;
}

// gradient = c(x_k) c'(x_k) 
Eigen::Vector3d SoftPointToPlaneConstraint::computeGradient(const Eigen::Vector3d& x) {
	Eigen::Vector3d gradient = (x.dot(normal) - plane_point.dot(normal)) * normal;
	return weight * gradient;
}

// hessian = c(x_k) c''(x_k) + c'(x_k)^T c'(x_k) 
Eigen::Matrix3d SoftPointToPlaneConstraint::computeHessian(const Eigen::Vector3d& x) {
	return weight * (normal * normal.transpose());
}


//// ConstraintManager Implementations

int ConstraintManager::getNumSoftConstraints() {
	return this->soft_constraints.size();
}

int ConstraintManager::getNumHardConstraints() {
	return this->hard_constraints.size();
}

void ConstraintManager::addHardConstraint(std::shared_ptr<HardConstraint> constraint) {
	this->hard_constraints.push_back(constraint);
}

void ConstraintManager::addHardConstraints(std::vector<std::shared_ptr<HardConstraint> > constraints_list) {
	this->hard_constraints.insert(this->hard_constraints.end(), constraints_list.begin(), constraints_list.end());
}

void ConstraintManager::addSoftConstraint(std::shared_ptr<SoftConstraint> constraint) {
	this->soft_constraints.push_back(constraint);
}

void ConstraintManager::addSoftConstraints(std::vector<std::shared_ptr<SoftConstraint> > constraints_list) {
	this->soft_constraints.insert(this->soft_constraints.end(), constraints_list.begin(), constraints_list.end());
}

const std::vector<std::shared_ptr<HardConstraint> >& ConstraintManager::getHardConstraints() {
	return this->hard_constraints;
}

const std::vector<std::shared_ptr<SoftConstraint> >& ConstraintManager::getSoftConstraints() {
	return this->soft_constraints;
}

// Needs to be called after energyfunction is created
void ConstraintManager::addHardCorrespondenceConstraints(TetrahedralMesh* source, TetrahedralMesh* target, const Eigen::MatrixXi& correspondences, Eigen::MatrixXd& subspace) {
	int num_source_verts = source->getNumVertices();
	int num_target_verts = target->getNumVertices();

	// Set correspondences as hard constraints
	// For hard constraints in the subspace we are moving the constrained vertices to the target position and setting DOF in subspace to 0
	Eigen::MatrixXd source_verts = source->getVertices();
	Eigen::MatrixXd target_verts = target->getVertices();
	for (int i = 0; i < correspondences.rows(); ++i) {
		int fixed_vert_idx = correspondences(i, 0);
		// Safety check whether the source correspondence vertex exists
		if (fixed_vert_idx >= num_source_verts || correspondences(i, 1) >= num_target_verts) {
			std::cerr << "The source and/or target correspondence vertex does not exist" << std::endl;
			return;
		}

		// Move source vertex to target vertex position
		source_verts.row(fixed_vert_idx) = target_verts.row(correspondences(i, 1));
		// Set DOF to 0 in subspace
		subspace.row(3 * fixed_vert_idx).setZero();
		subspace.row(3 * fixed_vert_idx + 1).setZero();
		subspace.row(3 * fixed_vert_idx + 2).setZero();

		// add as hard constraint to fix DOF for gradient and hessian calc
		this->addHardConstraint(std::shared_ptr<HardConstraint>(new HardFixedPointConstraint(fixed_vert_idx, target_verts.row(correspondences(i, 1)))));
	}
	source->setVertices(source_verts);
}

// Add correspondence constraints // Correspondence matrix: source_vertex, target_vertex
void ConstraintManager::addCorrespondenceConstraints(TetrahedralMesh* source, TetrahedralMesh* target, const Eigen::MatrixXi& correspondences, double correspondence_weight) {
	// Safety checks
	int num_source_verts = source->getNumVertices();

	// Set correspondences as soft constraints
	Eigen::MatrixXd target_verts = target->getVertices();
	for (int i = 0; i < correspondences.rows(); ++i) {
		int fixed_point_idx = correspondences(i, 0);
		// Safety check whether the source correspondence vertex exists
		if (fixed_point_idx >= num_source_verts) {
			std::cerr << "The source correspondence vertex does not exist" << std::endl;
			return;
		}

		Eigen::Vector3d target_pos = target_verts.row(correspondences(i, 1));
		this->addSoftConstraint(std::shared_ptr<SoftConstraint>(new SoftFixedPointConstraint(fixed_point_idx, target_pos, correspondence_weight)));
	}
}

void ConstraintManager::addFixedPointConstraints(TetrahedralMesh* tet_mesh, const std::vector<bool>& fixed_verts) {
	// Setting up the hard constraints (convert fixed_vertices to hard constraints)
	Eigen::MatrixXd verts = tet_mesh->getVertices();
	for (unsigned int i = 0; i < fixed_verts.size(); ++i) {
		if (fixed_verts[i]) {
			this->addHardConstraint(std::shared_ptr<HardConstraint>(new HardFixedPointConstraint(i, verts.row(i))));
		}
	}
}

// Overloading method for subspace version For the subspace it removes the Degrees of Freedom
void ConstraintManager::addFixedPointConstraints(TetrahedralMesh* tet_mesh, const std::vector<bool>& fixed_verts, Eigen::MatrixXd& subspace) {
	// Also add the regular fixed point constraints
	this->addFixedPointConstraints(tet_mesh, fixed_verts);

	Eigen::MatrixXd vertices = tet_mesh->getVertices();
	// Setting fixed vertices as zero in the subspace
	for (unsigned int i = 0; i < fixed_verts.size(); ++i) {
		if (fixed_verts[i]) {
			//this->addSoftConstraint(std::shared_ptr<SoftConstraint>(new SoftFixedPointConstraint(i, vertices.row(i), 10.0)));
			// remove DOF instead
			subspace.row(3 * i).setZero();
			subspace.row(3 * i + 1).setZero();
			subspace.row(3 * i + 2).setZero();
		}
	}
}

// Methods for clearing the constraints
void ConstraintManager::clearSoftConstraints() {
	this->soft_constraints.clear();
}

void ConstraintManager::clearHardConstraints() {
	this->hard_constraints.clear();
}

// Soft fixed frame constraint (point with attached frame get fixed)
std::vector<std::shared_ptr<SoftConstraint>> ConstraintManager::createSoftFixedFrameConstraint(int initial_constr_vert_idx, int target_constr_vert_idx, int num_frame_rings,
	const Eigen::Matrix3Xd& initial_verts_pos, const Eigen::Matrix3Xd& initial_verts_normals, const Eigen::Matrix3Xd& target_verts_pos,
	const Eigen::Matrix3Xd& target_verts_normals, const Eigen::SparseMatrix<double>& adjacency_matrix) {
	// Storage for all the softfixedpoint constraints
	std::vector<std::shared_ptr<SoftConstraint>> fixed_point_constraints;

	// Rigid translation for the frame
	Eigen::Vector3d initial_constr_vert_pos = initial_verts_pos.col(initial_constr_vert_idx);
	// TODO: see if more involved rigid transformation is required
	Eigen::Vector3d translation = target_verts_pos.col(target_constr_vert_idx) - initial_constr_vert_pos;

	// Get corresponding rotation for the involved vertex normals
	//https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
	Eigen::Vector3d n1 = initial_verts_normals.col(initial_constr_vert_idx);
	Eigen::Vector3d n2 = target_verts_normals.col(target_constr_vert_idx);
	Eigen::Matrix3d GG;
	GG << n1.dot(n2), -n1.cross(n2).norm(), 0,
		n1.cross(n2).norm(), n1.dot(n2), 0,
		0, 0, 1;
	Eigen::Matrix3d FFi;
	FFi << n1, (n2 - n1.dot(n2) * n1).normalized(), n2.cross(n1);
	Eigen::Matrix3d rotation = FFi * GG * FFi.inverse();

	std::vector< std::vector<int> > constrained_verts;
	for (int i = 0; i < num_frame_rings + 1; ++i) {
		constrained_verts.push_back(std::vector<int>());
	}
	constrained_verts[0].push_back(initial_constr_vert_idx);
	int num_constrained_verts = 1;


	// Weight for each depth
	double root_weight = 1.0;

	// Use Breadth First Search to get the ring of neighbours
	Eigen::VectorXd D, P;
	igl::bfs(adjacency_matrix, initial_constr_vert_idx, D, P);


	// Keep track of all seen vertices to ensure no double constraints are made
	std::set<int> visited_verts;
	visited_verts.insert(initial_constr_vert_idx);

	int current_depth = 1;
	std::set<int> current_depth_visited_verts;

	// Start at 1 skipping the root node
	for (int i = 1; i < D.size(); ++i) {
		int current_vertex_idx = D(i);

		// Check if the predecessor is in the set seen vertices
		//std::set<int>::iterator it = visited_verts.find(P(current_vertex_idx));
		if (!current_depth_visited_verts.empty() && visited_verts.find(P(current_vertex_idx)) == visited_verts.end()) {
			// Change the constraint weight for each subsequent depth
			//weight *= 0.5;
			//double distance_to_root_node = 1 - (static_cast<double>(current_depth) / (static_cast<double>(num_frame_rings) + 1.0));
			//double neighbour_weight = 3 * (distance_to_root_node * distance_to_root_node) - 2 * (distance_to_root_node * distance_to_root_node * distance_to_root_node);
			double distance_to_root_node = static_cast<double>(current_depth) / static_cast<double>(num_frame_rings);
			double neighbour_weight = exp(-(distance_to_root_node * distance_to_root_node));
			// Here the fixed point constraints can be made for the new depth
			for (std::set<int>::iterator it = current_depth_visited_verts.begin(); it != current_depth_visited_verts.end(); ++it) {
				// Use the distance in the initial configuration/frame to set the weight for the point constraint
				//double distance_to_root_node = (initial_verts_pos.col(*it) - initial_constr_vert_pos).norm();
				//fixed_point_constraints.push_back(std::shared_ptr<SoftConstraint>(new SoftFixedPointConstraint(*it, rotation * (initial_verts_pos.col(*it) + translation), neighbour_weight)));

				// For rigid alignment
				constrained_verts[current_depth].push_back(*it);
				++num_constrained_verts;
			}

			// Add current depth visited verts to the visited verts set to start looking at the next depth
			visited_verts.insert(current_depth_visited_verts.begin(), current_depth_visited_verts.end());
			current_depth_visited_verts.clear();
			++current_depth;
		}

		// If it has gone over all neighbours in the current rings break the loop
		if (current_depth > num_frame_rings) {
			break;
		}

		if (visited_verts.find(P(current_vertex_idx)) != visited_verts.end()) {
			// found
			current_depth_visited_verts.insert(current_vertex_idx);
		}
		else {
			// not found inside of seen_vertices, so that means it is either -1 or weird ordering??
			break;
		}
	}

	// for rigid alignment
	Eigen::MatrixXd constr_v1(num_constrained_verts, 3);
	Eigen::MatrixXd constr_v2(num_constrained_verts, 3);
	Eigen::MatrixXd constr_n2(num_constrained_verts, 3);
	int current_v = 0;
	for (unsigned int i = 0; i < constrained_verts.size(); ++i) {
		for (unsigned int j = 0; j < constrained_verts[i].size(); ++j) {
			int current_constrained_idx = constrained_verts[i][j];
			constr_v1.row(current_v) = initial_verts_pos.col(current_constrained_idx).transpose();
			constr_v2.row(current_v) = target_verts_pos.col(current_constrained_idx).transpose();
			constr_n2.row(current_v) = target_verts_normals.col(current_constrained_idx).normalized().transpose();
			++current_v;
		}
	}
	Eigen::Matrix3d actual_rotation;
	Eigen::RowVector3d actual_translation;
	igl::rigid_alignment(constr_v1, constr_v2, constr_n2, actual_rotation, actual_translation);

	// Create constraints
	for (unsigned int i = 0; i < constrained_verts.size(); ++i) {
		for (unsigned int j = 0; j < constrained_verts[i].size(); ++j) {
			int current_constrained_idx = constrained_verts[i][j];
			double distance_to_root_node = static_cast<double>(i) / static_cast<double>(num_frame_rings);
			double neighbour_weight = exp(-(distance_to_root_node * distance_to_root_node));
			fixed_point_constraints.push_back(std::shared_ptr<SoftConstraint>(new SoftFixedPointConstraint(current_constrained_idx, (initial_verts_pos.col(current_constrained_idx).transpose() * actual_rotation + actual_translation), neighbour_weight)));
		}
	}



	return fixed_point_constraints;
}
