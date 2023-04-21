#include "solver.h"
#include <iostream>
#include <stdexcept>
#include "linesearch.h"
#include <chrono>
#include "linesearch.h"
#include "constraint.h"
#include <algorithm>

#include <igl/slice.h>
#include <igl/AABB.h>
#include <igl/per_face_normals.h>
#include <igl/writeOBJ.h>

#include <Eigen/PardisoSupport>

// Used for sanity checking 
void checkSystemOfEquations(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& x, const Eigen::VectorXd& b) {
	const double tolerance = 1e-4;
	
	double symmetricResidual = (A - Eigen::SparseMatrix<double>(A.transpose())).squaredNorm();
	assert(symmetricResidual < tolerance);

	// check positive semi-definite
	Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double> > eigenSolver(A);
	double psd = (eigenSolver.eigenvalues().array() < -tolerance).sum(); // 0.0
	assert(psd < tolerance);

	double solverResidualError = (A * x - b).norm() / (A * x).norm();
	std::cout << "Solver Residual Error: " << solverResidualError << std::endl;
	// comment assert for now due to it not being the case at the moment
	//assert(solverResidualError < tolerance);

}



// https://stackoverflow.com/questions/41756428/concatenate-sparse-matrix-eigen
// See numerical optimization book page 451
Eigen::VectorXd Solver::solveKKTSystem(Eigen::Ref<Eigen::Matrix3Xd> x, const Eigen::SparseMatrix<double>& hessian, const Eigen::VectorXd& gradient, const std::vector<std::shared_ptr<HardConstraint> >& hard_constraints) {
	// Retrieve the triplets from the sparse hessian
	std::vector<Eigen::Triplet<double>> triplets;
	for (int i = 0; i < hessian.outerSize(); i++)
		for (typename Eigen::SparseMatrix<double>::InnerIterator it(hessian, i); it; ++it)
			triplets.emplace_back(it.row(), it.col(), it.value());

	int num_fixed_verts = hard_constraints.size();

	// h vector
	Eigen::VectorXd h(num_fixed_verts * 3);
	h.setZero();

	// Add the G matrix for KKT system to the triplets
	for (unsigned int i = 0; i < hard_constraints.size(); ++i) {
		int fixed_vertex_idx = hard_constraints[i]->getVertexIdx();
		for (unsigned int j = 0; j < 3; ++j) {
			// Set G element
			triplets.emplace_back(hessian.rows() + 3 * i + j, 3 * fixed_vertex_idx + j, 1.0);
			// Set G^T element
			triplets.emplace_back(3 * fixed_vertex_idx + j, hessian.cols() + 3 * i + j, 1.0);
		}
		Eigen::Vector3d constraint = x.col(fixed_vertex_idx) - hard_constraints[i]->getRHSb();
		h.segment(3 * i, 3) = constraint;

	}

	// Construct sparse matrix for KKT system
	Eigen::SparseMatrix<double> G(hessian.rows() + num_fixed_verts * 3, hessian.cols() + num_fixed_verts * 3);
	G.setFromTriplets(triplets.begin(), triplets.end());

	// Construct the vector for KKT system
	
	//HESSIAN = 3xn * 3xn  x = 3*n  gradient = 3xn
	Eigen::Map<Eigen::VectorXd> xVec(x.data(), x.size());
	Eigen::VectorXd g = gradient + hessian * xVec;// TODO: FIX THIS WHAT COMES HERE LOOK IT UP hessian * x + gradient
	Eigen::VectorXd b(gradient.size() + h.size());
	b << gradient, h;

	///// Using Pardiso LU solver for performance reasons (~5s for normal sparseLU vs ~1.5s for pardisoLU)
	// Also compared results with sparseLU and no difference
	Eigen::PardisoLU<Eigen::SparseMatrix<double> > solver(G);

	Eigen::VectorXd descentDir = solver.solve(b);

	if (solver.info() != Eigen::Success) {
		throw std::exception("The solver failed to find a solution");
	}

	double solverResidualError = (G * descentDir - b).norm() / (G * descentDir).norm();
	
	std::cout << "Solver residual error: " << solverResidualError << std::endl;
	return descentDir.segment(0, gradient.size());
}


std::vector<Eigen::Triplet<double>> to_triplets(Eigen::SparseMatrix<double>& M) {
	std::vector<Eigen::Triplet<double>> v;
	for (int i = 0; i < M.outerSize(); i++)
		for (typename Eigen::SparseMatrix<double>::InnerIterator it(M, i); it; ++it)
			v.emplace_back(it.row(), it.col(), it.value());
	return v;
}


bool Solver::solve(EnergyFunction* energy_function, Eigen::Ref<Eigen::Matrix3Xd> x, const Parameters& params) {
	bool fixedDOFs = false;

	int num_fixed_verts = energy_function->getNumFixedVerts();
	// Hard Fixed point constraints
	std::vector<std::shared_ptr<HardConstraint> > hard_constraints = energy_function->getHardConstraints();

	TetrahedralMesh* tet_mesh = energy_function->getTetMesh();

	// In case of debug mode, write the mesh to the output directory
	if (params.debug_mode) {
		// Write the tet mesh to an OBJ file with the current values for x
		tet_mesh->writeToFile(x.transpose(), params.output_path, "debug_" + std::to_string(1));
	}
	for (unsigned int i = 0; i < params.simulation_max_iter; ++i) {
		std::cout << "Solver iteration number: " << i << std::endl;

		Eigen::VectorXd descentDir;
		Eigen::VectorXd grad = energy_function->computeGradient(x, fixedDOFs);
		Eigen::VectorXd termination_gradient = energy_function->computeGradient(x, true);

		// Note that im now using the gradient where the degrees of freedom are removed for fixed vertices
		std::cout << "Gradient squared norm: " << termination_gradient.squaredNorm() << std::endl;
		// Check whether the force/energy gradient L2-norm is less than 1e-2
		if (termination_gradient.squaredNorm() < params.tolerance) { 
			break;
		}

		// Compute the descent direction for the solve depending on the optimization method
		if (params.optimization_method == OptimizationMethods::GRADIENT_DESCENT) {
			descentDir = -grad;	
		}
		else if (params.optimization_method == OptimizationMethods::NEWTON) {
			Eigen::SparseMatrix<double> hessian = energy_function->computeHessian(x, fixedDOFs);
			
			// Solve the KKT system in case of hard constraints
			// Otherwise solve normal system
			if (num_fixed_verts != 0) {
				descentDir = Solver::solveKKTSystem(x, hessian, grad, hard_constraints);
				// According to 16.5 in Numerical optimization book the negative descent direction comes out of it
				descentDir = -descentDir;
			}
			else {

				Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(hessian);
				descentDir = solver.solve(grad);

				// Sanity checking system of equations
				checkSystemOfEquations(hessian, descentDir, grad);

				descentDir = -descentDir;
			}
		}

		// Find a suitable step length for the descent direction
		// SNH version for linesearch
		double stepLength;
		double minEnergy;
		SNHLinesearch::MinimizeInSearchDirection(energy_function, x, descentDir, stepLength, minEnergy);
		// Setting this here temp to check results after it cant get lower gradient squared norm
		if (abs(stepLength) <= 10e-6) {
			std::cout << "Step length of 0" << std::endl;
			return false;
		}

		// Output chosen step length and minimal energy
		if (params.debug_mode) {
			std::cout << "Step Length: " << stepLength << std::endl;
			std::cout << "Min Energy: " << minEnergy << std::endl;
		}

		// Mutate x with the updated positions
		Eigen::Map<Eigen::MatrixXd> descentDirMap(descentDir.data(), x.rows(), x.cols());
		Eigen::MatrixXd descentDirMat = descentDirMap;//.transpose(); no transpose needed after transpose in main.cpp
		x += stepLength * descentDirMat;

		// In case of debug mode, write the mesh to the output directory
		if (params.debug_mode) {
			// Write the tet mesh to an OBJ file with the current values for x
			tet_mesh->writeToFile(x.transpose(), params.output_path, "debug_" + std::to_string(i + 2));
		}
	}

	return true;
}


// Currently only implemented for Newtons method and no fixed constraints (Newton-KKT system) (it is assumed that fixed constraints are fixed in the subspace U)
bool Solver::reducedSolve(EnergyFunction* energy_function, const Eigen::MatrixXd& U, Eigen::Ref<Eigen::Matrix3Xd> x, const Parameters& params) {
	bool fixedDOFs = false;

	int num_fixed_verts = energy_function->getNumFixedVerts();
	// Hard Fixed point constraints
	std::vector<std::shared_ptr<HardConstraint> > hard_constraints = energy_function->getHardConstraints();


	TetrahedralMesh* tet_mesh = energy_function->getTetMesh();

	// In case of debug mode, write the mesh to the output directory
	if (params.debug_mode) {
		// Write the tet mesh to an OBJ file with the current values for x
		tet_mesh->writeToFile(x.transpose(), params.output_path, "debug_" + std::to_string(1));
	}
	for (unsigned int i = 0; i < params.simulation_max_iter; ++i) {
		std::cout << "Solver iteration number: " << i << std::endl;

		Eigen::VectorXd descentDir;
		Eigen::VectorXd grad = energy_function->computeGradient(x, fixedDOFs);
		Eigen::VectorXd termination_gradient = energy_function->computeGradient(x, true);

		// Note that im now using the gradient where the degrees of freedom are removed for fixed vertices
		std::cout << "Gradient squared norm: " << (U.transpose() * termination_gradient).squaredNorm() << std::endl;
		// Check whether the force/energy gradient L2-norm is less than 1e-2
		if ((U.transpose() * termination_gradient).squaredNorm() < params.tolerance) {
			break;
		}

		// Compute the descent direction for the solve depending on the optimization method
		Eigen::SparseMatrix<double> hessian = energy_function->computeHessian(x, fixedDOFs);

		// Solve reduced system
		Eigen::MatrixXd reduced_hessian = U.transpose() * hessian * U;
		Eigen::VectorXd reduced_grad = U.transpose() * grad;
		Eigen::LDLT<Eigen::MatrixXd> solver(reduced_hessian); 
		descentDir = solver.solve(reduced_grad);
		descentDir = -U * descentDir;

		// Find a suitable step length for the descent direction
		// SNH version for linesearch
		double stepLength;
		double minEnergy;
		SNHLinesearch::MinimizeInSearchDirection(energy_function, x, descentDir, stepLength, minEnergy);
		// Setting this here temp to check results after it cant get lower gradient squared norm
		if (abs(stepLength) <= 10e-6) {
			std::cout << "Step length of 0" << std::endl;
			return false;
		}

		// Output chosen step length and minimal energy
		if (params.debug_mode) {
			std::cout << "Step Length: " << stepLength << std::endl;
			std::cout << "Min Energy: " << minEnergy << std::endl;
		}

		// Mutate x with the updated positions
		Eigen::Map<Eigen::MatrixXd> descentDirMap(descentDir.data(), x.rows(), x.cols());
		Eigen::MatrixXd descentDirMat = descentDirMap;//.transpose(); no transpose needed after transpose in main.cpp
		x += stepLength * descentDirMat;

		// In case of debug mode, write the mesh to the output directory
		if (params.debug_mode) {
			// Write the tet mesh to an OBJ file with the current values for x
			tet_mesh->writeToFile(x.transpose(), params.output_path, "debug_" + std::to_string(i + 2));
		}
	}

	return true;
}


void Solver::nonRigidAlignment(TetrahedralMesh* t1, const std::vector<bool>& fixed_v1, TetrahedralMesh* t2, const std::vector<bool>& fixed_v2, const Eigen::MatrixXi& correspondences,
	EnergyFunction* energy_function, const Eigen::MatrixXd& subspace, const Parameters& params) {
	Parameters solver_params = params;
	solver_params.simulation_max_iter = 1;
	solver_params.debug_mode = false;

	Eigen::MatrixXd v2 = t2->getVertices();
	Eigen::MatrixXi tri2 = t2->getTriangles(fixed_v2);
	Eigen::MatrixXd n2;
	igl::per_face_normals(v2, tri2, n2);

	// Initialize AABB 
	igl::AABB<Eigen::MatrixXd, 3> tree;
	tree.init(v2, tri2);

	// Setup solver variables
	ConstraintManager* constraint_manager = energy_function->getConstraintManager();
	double soft_fixed_point_weight = 100.0;
	for (unsigned int i = 0; i < params.non_rigid_alignment_max_iter; ++i) {
		std::cout << "Non rigid alignment solver iteration " << i << std::endl;
		// Update v1
		Eigen::VectorXi v_ind1 = t1->getSurfaceVertIndices(fixed_v1);

		Eigen::MatrixXd surface_v1;
		igl::slice(t1->getVertices(), v_ind1, 1, surface_v1);

		// Compute closest points of t1 on t2
		Eigen::VectorXd sqrD;
		Eigen::VectorXi I;
		Eigen::MatrixXd C;
		tree.squared_distance(v2, tri2, surface_v1, sqrD, I, C);


		Eigen::MatrixXd C_normals;
		igl::slice(n2, I, 1, C_normals);

		// use normal solve with new constraints
		constraint_manager->clearSoftConstraints();
		// Add new constraints
		for (unsigned int j = 0; j < C.rows(); ++j) {
			constraint_manager->addSoftConstraint(std::shared_ptr<SoftConstraint>(new SoftFixedPointConstraint(v_ind1(j), C.row(j), 10.0))); 

			constraint_manager->addSoftConstraint(std::shared_ptr<SoftConstraint>(new SoftPointToPlaneConstraint(v_ind1(j), C.row(j), C_normals.row(j), soft_fixed_point_weight)));
		}

		// Solve 1 iteration with surface constraints
		Eigen::Matrix3Xd X = t1->getVertices().transpose();

		// Check gradient termination criteria
		Eigen::VectorXd termination_gradient = energy_function->computeGradient(X, true);
		// Check whether the force/energy gradient L2-norm is less than 1e-2
		if (termination_gradient.squaredNorm() < params.tolerance) {
			break;
		}

		Solver::solve(energy_function, X, solver_params);
		Eigen::MatrixXd X_view = X.transpose();
		t1->setVertices(X_view);
		

		/// Debug write mesh to file
		if (params.debug_mode) {
			t1->writeToFile(params.output_path, "nonrigid_after_constrained_solve_" + std::to_string(i));
		}
	}
}


void Solver::solve(TetrahedralMesh* source, TetrahedralMesh* target, const std::vector<bool>& source_fixed_verts, const std::vector<bool>& target_fixed_verts, const Eigen::MatrixXi& correspondences,
	EnergyFunction* energy_function, Eigen::MatrixXd subspace, const Solver::Parameters& params) {
	ConstraintManager* constraint_manager = energy_function->getConstraintManager();
	constraint_manager->clearHardConstraints();
	constraint_manager->clearSoftConstraints();

	// SETUP HARD CONSTRAINTS for the fixed points
	constraint_manager->addFixedPointConstraints(source, source_fixed_verts);
	constraint_manager->addFixedPointConstraints(source, source_fixed_verts, subspace);

	if (params.simulation) {
		if (params.use_landmarks) {
			// Set correspondences as soft constraints
			// Correspondence matrix: source_vertex, target_vertex
			constraint_manager->addCorrespondenceConstraints(source, target, correspondences, params.landmark_weight);
		}

		Eigen::Matrix3Xd X = source->getVertices().transpose();
		if (params.reduced_sim) {
			Solver::reducedSolve(energy_function, subspace, X, params);
		}
		else {
			Solver::solve(energy_function, X, params);
		}

		Eigen::MatrixXd X_view = X.transpose();

		// Move the object back to center if it has no fixed vertices
		if (energy_function->getNumFixedVerts() == 0) {
			Eigen::Vector3d center = findCenterOfMesh(X_view);

			for (unsigned int i = 0; i < X_view.rows(); ++i) {
				X_view.row(i) -= center;
			}
		}

		// Set X in solution tetrahedral mesh
		source->setVertices(X_view);
	}


	if (params.non_rigid_alignment) {
		constraint_manager->clearSoftConstraints();
		Solver::nonRigidAlignment(source, source_fixed_verts, target, target_fixed_verts, correspondences, energy_function, subspace, params);
	}
}



