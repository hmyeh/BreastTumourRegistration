#ifndef SOLVER_H
#define SOLVER_H

#include <string>
#include <Eigen/Dense>

#include "energy.h"
#include "tetrahedralmesh.h"
#include "optimizationmethods.h"



struct SolverParams {

};

/*
Solver class to hold the energy function and peform Newton's method combined with linesearch
*/
class Solver {
public:
	// Struct containing parameters for solve
	struct Parameters {
		bool simulation = true;
		bool non_rigid_alignment = true;
		double tolerance = 10e-4;
		int simulation_max_iter = 100;
		int non_rigid_alignment_max_iter = 20;
		OptimizationMethods optimization_method = OptimizationMethods::NEWTON;
		bool reduced_sim = true;
		bool use_landmarks = true;
		double landmark_weight = 100.0;
		bool debug_mode = true;
		std::string output_path = "output/";

		Parameters() {}
		Parameters(bool simulation, bool non_rigid_alignment, bool reduced_sim, bool use_landmarks, bool debug_mode) : simulation(simulation), non_rigid_alignment(non_rigid_alignment), reduced_sim(reduced_sim), use_landmarks(use_landmarks), debug_mode(debug_mode) {}
	};

private:
	static Eigen::VectorXd solveKKTSystem(Eigen::Ref<Eigen::Matrix3Xd> x, const Eigen::SparseMatrix<double>& hessian, const Eigen::VectorXd& gradient, const std::vector<std::shared_ptr<HardConstraint> >& hard_constraints);
	// The method to solve the energy function limited by a max iteration number where x is the deformed vertices
	// Static version so give energy function and the x you want to test with (energy function should contain tetrahedral mesh)
	// Made it bool where true means it fully converged, and false that it did not (still might be a valid result) (added for breasts fixed points to know when it is just impossible to progress)

public:
	static bool solve(EnergyFunction* energy_function, Eigen::Ref<Eigen::Matrix3Xd> x, const Parameters& params);

	// reduced subspace simulation (U = skinning space)
	static bool reducedSolve(EnergyFunction* energy_function, const Eigen::MatrixXd& U, Eigen::Ref<Eigen::Matrix3Xd> x, const Parameters& params);

	static void nonRigidAlignment(TetrahedralMesh* t1, const std::vector<bool>& fixed_v1, TetrahedralMesh* t2, const std::vector<bool>& fixed_v2, const Eigen::MatrixXi& correspondences,
		EnergyFunction* energy_function, const Eigen::MatrixXd& subspace, const Parameters& params);

	// Calling the full method just method...
	// using copy of subspace because we are defining constraints inside
	// simulation refers to the first simulation... better naming required
	static void solve(TetrahedralMesh* source, TetrahedralMesh* target, const std::vector<bool>& source_fixed_verts, const std::vector<bool>& target_fixed_verts, const Eigen::MatrixXi& correspondences,
		EnergyFunction* energy_function, Eigen::MatrixXd subspace, const Solver::Parameters& params);

	//Prediction-correction iterative scheme for stress-free geometry estimation
	//static void computeStressFreeGeometry(EnergyFunction* energy_function, Eigen::Ref<Eigen::Matrix3Xd> x, OptimizationMethods optimizationMethod, unsigned int max_iter = 5, bool debug = false, double tolerance = 10e-4);
};

#endif
