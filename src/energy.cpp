#include "energy.h"
#include "deformationgradient.h"
#include <iostream>
#include <math.h> 
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <Eigen/Eigenvalues> 

// Elastic Energy 
double ElasticEnergy::computeEnergy(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x) {
	// Create vector to store the elastic energy for each element
	Eigen::VectorXd element_energies(elements.size());

	// Sum the strain energy for each element multiplied with the weight (volume)
	// In this case also find the deformed vertex positions corresponding for each element
	// Using integer loop for OpenMP
	//#pragma omp parallel for
	for (int i = 0; i < elements.size(); ++i) {
		auto t1 = std::chrono::high_resolution_clock::now();
		Element* current_element = elements[i].get();
		// Get correct vertices for tetrahedron/element
		std::vector<int> indices = current_element->getIndices();
		std::vector<Eigen::VectorXd> element_verts(4);
		for (unsigned int j = 0; j < indices.size(); ++j) {
			element_verts[j] = x.col(indices[j]);
		}

		// The deformation total_gradient object storing all info about F and its invariants
		DeformationGradient deformation_gradient = current_element->computeDeformationGradient(element_verts);

		// Use the deformation total_gradient as input to compute the strain energy
		double tet_strain_energy = current_element->computeStrainEnergy(deformation_gradient);
		element_energies[i] = current_element->getWeight() * tet_strain_energy;
	}

	// Summing the element energy entries
	double total_energy = element_energies.sum();
	return total_energy;
}


Eigen::VectorXd ElasticEnergy::computeGradient(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x) {
	int num_verts = x.cols();
	// The global total_gradient vector with the assumption of using XYZ coordinates (Maybe change it so 2D could also be possible)
	Eigen::VectorXd total_gradient(num_verts * 3); // At the moment I am always assuming XYZ coordinates... maybe should change that if i wanna test 2d stuff...
	total_gradient.setZero();

	// Store each element total_gradient in the vector for OpenMP writing
	std::vector<Eigen::VectorXd> element_gradients(elements.size());

	// Compute all element gradients with OpenMP multithreading
	#pragma omp parallel for
	for (int i = 0; i < elements.size(); ++i) {
		Element* current_element = elements[i].get();
		// Get correct vertices for tetrahedron/element
		std::vector<int> indices = current_element->getIndices();
		std::vector<Eigen::VectorXd> element_verts(4);
		for (unsigned int j = 0; j < indices.size(); ++j) {
			element_verts[j] = x.col(indices[j]);
		}
		DeformationGradient deformation_gradient = current_element->computeDeformationGradient(element_verts);

		// volume/weight * total_gradient
		Eigen::VectorXd grad = current_element->getWeight() * current_element->computeStrainGradient(deformation_gradient);
		element_gradients[i] = grad;
	}

	// Sum all computed elemental gradients (No OpenMP)
	for (int i = 0; i < elements.size(); ++i) {
		// Get correct vertices for tetrahedron/element
		std::vector<int> indices = elements[i]->getIndices();

		// Sum each vertex segment of the total_gradient into the total total_gradient
		for (std::vector<int>::size_type j = 0; j != indices.size(); j++) {
			int vertexIndex = indices[j];
			total_gradient.segment(vertexIndex * 3, 3) += element_gradients[i].segment(j * 3, 3);
		}
	}

	return total_gradient;
}


Eigen::SparseMatrix<double> ElasticEnergy::computeHessian(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x) {
	int num_verts = x.cols();
	// triplet list to insert/sum in the sparse matrix (estimated to be number of tets x 12x12 matrices)
	int estimated_triplet_list_size = elements.size() * 144;
	std::vector<Eigen::Triplet<double> > triplet_list;
	triplet_list.reserve(estimated_triplet_list_size);

	std::vector<Eigen::MatrixXd> element_hessians(elements.size());

	// Sum all the element Hessians into the global Hessian matrix (currently done through each individual entry)
	#pragma omp parallel for
	for (int i = 0; i < elements.size(); ++i) {
		Element* current_element = elements[i].get();
		std::vector<int> indices = current_element->getIndices();
		std::vector<Eigen::VectorXd> element_verts(4);
		for (unsigned int j = 0; j < indices.size(); ++j) {
			element_verts[j] = x.col(indices[j]);
		}
		DeformationGradient deformation_gradient = current_element->computeDeformationGradient(element_verts);
		Eigen::MatrixXd hessian = current_element->getWeight() * current_element->computeStrainHessian(deformation_gradient);
		element_hessians[i] = hessian;
	}

	// Assemble global matrix
	for (int e = 0; e < element_hessians.size(); ++e) {
		Eigen::MatrixXd hessian = element_hessians[e];
		std::vector<int> indices = elements[e]->getIndices();
		for (unsigned int i = 0; i < hessian.rows(); ++i) {
			// Compute the global Row index
			int row_rest_index = i % 3;
			int row_tet_index = floor(((double)i) / 3.0); // 3 as in XYZ coordinates 3.0 for double
			int row_index = indices[row_tet_index] * 3 + row_rest_index;
			for (unsigned int j = 0; j < hessian.cols(); ++j) {
				// Compute the global Col index
				int col_rest_index = j % 3;
				int col_tet_index = floor(((double)j) / 3.0); // 3 as in XYZ coordinates 3.0 for double
				int col_index = indices[col_tet_index] * 3 + col_rest_index;

				// If it is a nonzero entry (smaller than 1e-4), then insert it into the triplet_list
				if (abs(hessian(i, j)) >= 1e-4) {
					triplet_list.push_back(Eigen::Triplet<double>(row_index, col_index, hessian(i, j)));
				}
			}
		}
	}


	// Construct sparse matrix
	Eigen::SparseMatrix<double> total_hessian(num_verts * 3, num_verts * 3);
	total_hessian.reserve(triplet_list.size());
	total_hessian.setFromTriplets(triplet_list.begin(), triplet_list.end());

	return total_hessian;
}



// Gravitational Energy

double GravitationalEnergy::computeEnergy(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x, const Eigen::Vector3d& g) {
	// Compute the gradient which is the -force of gravity
	Eigen::VectorXd gradient = GravitationalEnergy::computeGradient(elements, x, g);
	Eigen::Map<const Eigen::VectorXd> x_vec(x.data(), x.cols() * x.rows());
	// dot product should be fine since all other entries outside of the y axis should be 0.0
	return gradient.dot(x_vec);
}

Eigen::VectorXd GravitationalEnergy::computeGradient(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x, const Eigen::Vector3d& g) {
	int num_verts = x.cols();
	Eigen::VectorXd total_gradient(num_verts * 3);
	total_gradient.setZero();

	// The inv_num_element_verts is assuming it is a tetrahedron
	double inv_num_element_verts = 1.0 / 4.0;

	for (auto it = elements.begin(); it != elements.end(); ++it) { //std::vector<std::unique_ptr<Element> >::iterator
		// Get the weight/volume of the element tet
		double weight = (*it)->getWeight();
		double density = (*it)->getDensity();
		double mass = density * weight;

		// Get the indices of the verts of the tet/element
		std::vector<int> indices = (*it)->getIndices();

		for (std::vector<int>::iterator indexIt = indices.begin(); indexIt != indices.end(); ++indexIt) {
			// +1 is assuming the y coordinate is the direction of gravity
			total_gradient.segment(3 * (*indexIt), 3) += inv_num_element_verts * mass * g;
		}

	}

	return total_gradient;
}

Eigen::SparseMatrix<double> GravitationalEnergy::computeHessian(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x, const Eigen::Vector3d& g) {
	int num_verts = x.cols();
	// Note that the hessian is 0.0
	return Eigen::SparseMatrix<double>(num_verts * 3, num_verts * 3);
}


//// Penalty Energy (Soft Constraints)
double PenaltyEnergy::computeEnergy(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x, const std::vector<std::shared_ptr<SoftConstraint> >& constraints, double weight) {
	double total_energy = 0.0;
	for (auto it = constraints.begin(); it != constraints.end(); ++it) {
		int vertex_idx = (*it)->getVertexIdx();
		total_energy += (*it)->computeConstraint(x.col(vertex_idx));
	}

	// Multiply summed constraint energies with weight
	total_energy *= weight;
	return total_energy;
}

Eigen::VectorXd PenaltyEnergy::computeGradient(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x, const std::vector<std::shared_ptr<SoftConstraint> >& constraints, double weight) {
	int num_verts = x.cols();
	Eigen::VectorXd total_gradient(num_verts * 3);
	total_gradient.setZero();

	for (auto it = constraints.begin(); it != constraints.end(); ++it) {
		int vertex_idx = (*it)->getVertexIdx();
		total_gradient.segment(vertex_idx * 3, 3) += (*it)->computeGradient(x.col(vertex_idx));
	}

	// Multiply by the weight
	total_gradient *= weight;
	return total_gradient;

}

Eigen::SparseMatrix<double> PenaltyEnergy::computeHessian(const std::vector<std::unique_ptr<Element> >& elements, const Eigen::Matrix3Xd& x, const std::vector<std::shared_ptr<SoftConstraint> >& constraints, double weight) {
	int num_verts = x.cols();
	// triplet list to insert/sum in the sparse matrix (estimated to be number of tets x 12x12 matrices)
	//std::vector<std::shared_ptr<SoftConstraint> > soft_constraints = this->constraint_manager->getSoftConstraints();
	int estimated_triplet_list_size = constraints.size() * 9;
	std::vector<Eigen::Triplet<double> > triplet_list;
	triplet_list.reserve(estimated_triplet_list_size);

	std::vector<Eigen::MatrixXd> constraint_hessians(constraints.size());

	// Sum all the element Hessians into the global Hessian matrix (currently done through each individual entry)
//#pragma omp parallel for
	for (auto it = constraints.begin(); it != constraints.end(); ++it) {
		int vertex_idx = (*it)->getVertexIdx();
		Eigen::MatrixXd hessian = (*it)->computeHessian(x.col(vertex_idx));
		for (unsigned int i = 0; i < hessian.rows(); ++i) {
			for (unsigned int j = 0; j < hessian.cols(); ++j) {
				if (abs(hessian(i, j)) >= 1e-4) {
					triplet_list.push_back(Eigen::Triplet<double>(vertex_idx * 3 + i, vertex_idx * 3 + j, weight * hessian(i, j))); //this->weight * hessian(i, j)
				}
			}
		}
	}

	// Construct sparse matrix
	Eigen::SparseMatrix<double> total_hessian(num_verts * 3, num_verts * 3);
	total_hessian.reserve(triplet_list.size());
	total_hessian.setFromTriplets(triplet_list.begin(), triplet_list.end());

	return total_hessian;
}



// EnergyFunction method implementations
EnergyFunction::EnergyFunction(TetrahedralMesh* tet_mesh, std::vector<std::unique_ptr<Element::Parameters> > area_params, double penalty_weight, bool gravity_on, Eigen::Vector3d gravity) : tet_mesh(tet_mesh),
	penalty_weight(penalty_weight), gravity_on(gravity_on), gravity(gravity), area_params(std::move(area_params)), constraint_manager(ConstraintManager()) {
	// Get the transposed version of the verts and tets
	Eigen::MatrixXd verts = tet_mesh->getVertices().transpose();
	Eigen::MatrixXi tets = tet_mesh->getTetrahedrons().transpose();

	// Create Finite Elements here
	this->createElements();

	// Set num vertices (number of cols 3xn matrix)
	this->num_verts = tet_mesh->getNumVertices();
}


// Material_models per area
void EnergyFunction::createElements() {
	// Get the transposed version of the verts and tets
	Eigen::Matrix3Xd verts = tet_mesh->getVertices().transpose();
	Eigen::Matrix4Xi tets = tet_mesh->getTetrahedrons().transpose();

	std::vector<std::vector<int> > areas = tet_mesh->getAreas();

	// Safety check that areas and element_params are the same size
	assert(areas.size() == area_params.size());

	// Setup the list of elements for FEM
	int num_tets = tets.cols();
	this->elements.resize(num_tets);

	int current_element_idx = 0;
	for (int i = 0; i < static_cast<int>(areas.size()); ++i) {
		//std::vector<int> area = areas[i];
		for (int j = 0; j < static_cast<int>(areas[i].size()); ++j) {
			int tet_idx = areas[i][j];
			// Setup the required indices and vertex rows required for the element
			Eigen::MatrixXi tet = tets.col(tet_idx);
			std::vector<Eigen::VectorXd> x(tet.size());
			for (int k = 0; k < static_cast<int>(tet.size()); ++k)
				x[k] = verts.col(tet(k));

			std::vector<int> tet_vec(tet.data(), tet.data() + tet.size());

			// Create new Tetrahedron and add to elements std::vector
			this->elements[current_element_idx] = std::unique_ptr<Element>(new Tetrahedron(x, tet_vec, area_params[i].get()));
			//this->elements[current_element_idx] = element;

			// Update current element index
			++current_element_idx;
		}
	}
}


double EnergyFunction::computeEnergy(const Eigen::Matrix3Xd& x) {
	// Sanity check same number of vertices
	assert(this->num_verts == x.cols());

	double total_energy = ElasticEnergy::computeEnergy(elements, x);

	// Add the gravitational energy if it is needed
	if (gravity_on) {
		total_energy += GravitationalEnergy::computeEnergy(elements, x, gravity);
	}

	// Add the penalty energy function
	total_energy += PenaltyEnergy::computeEnergy(elements, x, constraint_manager.getSoftConstraints(), penalty_weight);

	return total_energy;
}


Eigen::VectorXd EnergyFunction::computeGradient(const Eigen::Matrix3Xd& x, bool fixedDOFs) {
	// Sanity check same number of vertices
	assert(this->num_verts == x.cols());

	// Setup vector for the total gradient
	Eigen::VectorXd total_gradient = ElasticEnergy::computeGradient(elements, x);//(this->num_verts * 3);
	//total_gradient.setZero();

	// Add the gravitational gradient if it is needed
	if (gravity_on) {
		total_gradient += GravitationalEnergy::computeGradient(elements, x, gravity);
	}

	// Add the penalty gradient term
	total_gradient += PenaltyEnergy::computeGradient(elements, x, constraint_manager.getSoftConstraints(), penalty_weight);

	std::vector<std::shared_ptr<HardConstraint> > hard_constraints = this->constraint_manager.getHardConstraints();

	// Any "postprocessing" to the gradients such as removing DOF
	// For fixed vertices set the total_gradient to 0
	if (fixedDOFs) {
		for (unsigned int j = 0; j < hard_constraints.size(); ++j) {
			int fixed_vertex_idx = hard_constraints[j]->getVertexIdx();
			// set xyz for vertex in total_gradient to 0
			total_gradient(3 * fixed_vertex_idx) = 0;
			total_gradient(3 * fixed_vertex_idx + 1) = 0;
			total_gradient(3 * fixed_vertex_idx + 2) = 0;
		}
	}

	return total_gradient;
}

Eigen::SparseMatrix<double> EnergyFunction::computeHessian(const Eigen::Matrix3Xd& x, bool fixedDOFs) {
	// Sanity check same number of vertices
	assert(this->num_verts == x.cols());

	// Setup sparse matrix for the total hessian
	Eigen::SparseMatrix<double> total_hessian = ElasticEnergy::computeHessian(elements, x);//(this->num_verts * 3, this->num_verts * 3);
	//total_hessian.setZero();

	// The gravitational Hessian is constant 0, so could be not added at all...
	if (gravity_on) {
		total_hessian += GravitationalEnergy::computeHessian(elements, x, gravity);
	}

	// Add the penalty Hessian term
	total_hessian += PenaltyEnergy::computeHessian(elements, x, constraint_manager.getSoftConstraints(), penalty_weight);

	std::vector<std::shared_ptr<HardConstraint> > hard_constraints = this->constraint_manager.getHardConstraints();

	// Any "postprocessing" to the gradients such as removing DOF
	if (fixedDOFs) {
		for (unsigned int j = 0; j < hard_constraints.size(); ++j) {
			int fixed_vertex_idx = hard_constraints[j]->getVertexIdx();

			// set xyz for vertex in total_hessian to 0 and diagonal to 1
			total_hessian.col(3 * fixed_vertex_idx) *= 0;
			total_hessian.col(3 * fixed_vertex_idx + 1) *= 0;
			total_hessian.col(3 * fixed_vertex_idx + 2) *= 0;
		}

		Eigen::SparseMatrix<double> total_hessian_transposed = total_hessian.transpose();
		for (unsigned int j = 0; j < hard_constraints.size(); ++j) {
			int fixed_vertex_idx = hard_constraints[j]->getVertexIdx();

			// set xyz for vertex in total_hessian to 0 and diagonal to 1
			total_hessian_transposed.col(3 * fixed_vertex_idx) *= 0;
			total_hessian_transposed.col(3 * fixed_vertex_idx + 1) *= 0;
			total_hessian_transposed.col(3 * fixed_vertex_idx + 2) *= 0;

			// Set diagonal to 1
			total_hessian_transposed.coeffRef(3 * fixed_vertex_idx, 3 * fixed_vertex_idx) = 1.0;
			total_hessian_transposed.coeffRef(3 * fixed_vertex_idx + 1, 3 * fixed_vertex_idx + 1) = 1.0;
			total_hessian_transposed.coeffRef(3 * fixed_vertex_idx + 2, 3 * fixed_vertex_idx + 2) = 1.0;
		}
		total_hessian = total_hessian_transposed.transpose();
	}

	return total_hessian;
}


Eigen::SparseMatrix<double> EnergyFunction::getMassMatrix() {

	// triplet list to insert/sum in the sparse matrix
	double estimated_triplet_list_size = this->num_verts * 3 * 4;
	std::vector<Eigen::Triplet<double> > triplet_list;
	triplet_list.reserve(estimated_triplet_list_size);

	// The inv_num_element_verts is assuming it is a tetrahedron
	double inv_num_element_verts = 1.0 / 4.0;

	for (std::vector<std::unique_ptr<Element> >::iterator it = this->elements.begin(); it != this->elements.end(); ++it) {
		// Get the weight/volume of the element tet
		double weight = (*it)->getWeight();
		double density = (*it)->getDensity();
		double mass = density * weight;

		// Get the indices of the verts of the tet/element
		std::vector<int> indices = (*it)->getIndices();
		for (std::vector<int>::iterator indexIt = indices.begin(); indexIt != indices.end(); ++indexIt) {
			triplet_list.push_back(Eigen::Triplet<double>(3 * (*indexIt), 3 * (*indexIt), inv_num_element_verts * mass));
			triplet_list.push_back(Eigen::Triplet<double>(3 * (*indexIt) + 1, 3 * (*indexIt) + 1, inv_num_element_verts * mass));
			triplet_list.push_back(Eigen::Triplet<double>(3 * (*indexIt) + 2, 3 * (*indexIt) + 2, inv_num_element_verts * mass));
		}

	}


	// Construct sparse matrix
	Eigen::SparseMatrix<double> massMatrix(this->num_verts * 3, this->num_verts * 3);
	massMatrix.setFromTriplets(triplet_list.begin(), triplet_list.end());

	return massMatrix;
}

double EnergyFunction::computeElasticEnergy(const Eigen::Matrix3Xd& x) {
	// Sanity check same number of vertices
	assert(this->num_verts == x.cols());

	return ElasticEnergy::computeEnergy(elements, x);
}

// EnergyFunction Getter implementations
int EnergyFunction::getNumFixedVerts() { return this->constraint_manager.getHardConstraints().size(); }
const std::vector<std::shared_ptr<HardConstraint> >& EnergyFunction::getHardConstraints() { return this->constraint_manager.getHardConstraints(); }

TetrahedralMesh* EnergyFunction::getTetMesh() { return this->tet_mesh; }

ConstraintManager* EnergyFunction::getConstraintManager() { return &constraint_manager; }

const std::vector<std::unique_ptr<Element::Parameters> >& EnergyFunction::getMaterialParams() { return this->area_params; }

// EnergyFunction Setter implementations
void EnergyFunction::setGravityOn(bool gravity_on) { this->gravity_on = gravity_on; }

void EnergyFunction::setGravityVector(Eigen::Vector3d gravity) {
	this->gravity = gravity; 
	//this->gravitational_energy.setGravitationalConstant(gravity);
}

Eigen::Vector3d EnergyFunction::getGravity() { return this->gravity; }
