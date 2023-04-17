#ifndef SCENE_H
#define SCENE_H

#include <string>

#include <Eigen/Dense>

#include "solver.h"
#include "tetrahedralmesh.h"
#include "energy.h"


class Scene {
private:
	TetrahedralMesh source;
	TetrahedralMesh target; //  might need to change it from tetrahedralmesh to a triangle mesh...
	// Create additional tet mesh for deformation
	TetrahedralMesh solution;

	ImplicitGeometry implicit_solution;
	TriangleMesh implicit_target;

	std::vector<bool> source_fixed_verts;
	std::vector<bool> target_fixed_verts; // this is for if the target is a volume and not simply the surface of the breast

	Eigen::MatrixXi correspondences;
	int num_samples_subspace;
	Eigen::MatrixXd subspace;

	double E, v, density;

	EnergyFunction energy_function;
	
public:
	Scene() {}

	// MISSING Implicitgeometry and target tumour
	Scene(const TetrahedralMesh& source, const std::vector<bool>& source_fixed_verts, int num_sample_verts_subspace, bool gravity_on, Eigen::Vector3d gravity, double E, double v, double density, double penalty_weight);

	Scene(const TetrahedralMesh& source, const TetrahedralMesh& target, const std::vector<bool>& source_fixed_verts, const std::vector<bool>& target_fixed_verts, const Eigen::MatrixXi& correspondences,
		int num_sample_verts_subspace, bool gravity_on, Eigen::Vector3d gravity, double E, double v, double density, double penalty_weight);


	Scene(std::string data_folder, std::string source_file, std::string target_file, std::string source_fixed_verts_file, std::string target_fixed_verts_file, std::string source_tumour_file, std::string target_tumour_file, std::string correspondences_file,
		bool rigid_alignment, int num_sample_verts_subspace, bool gravity_on, Eigen::Vector3d gravity, double E, double v, double density, double penalty_weight);


	void simulate(Solver::Parameters solver_params = Solver::Parameters(), std::string file_name = "solution");

	// THIS FUNCTION ONLY WORKS FOR EXPERIMENTS
	void evaluate(const std::map<std::string, ImplicitGeometry>& tumour_maps, std::string output_path);
	void evaluate(const ImplicitGeometry& implicit_geometry, std::string output_path, std::string file_name = "scores.csv");


	// Method to allow for different tumour choices
	void evaluate(std::string output_path = "output/", std::string file_name = "implicit_result");

	// GETTERS FOR INFORMATION TO SHOW THE MESH AND EDIT PROPERTIES...

	void changeMaterialParams(double E, double v, double density);
	void changeGravitationalDirection(Eigen::Vector3d gravity);
	void setGravityOn(bool gravity_on);
	void setCorrespondences(const Eigen::MatrixXi& correspondences);

	void reset();

	TetrahedralMesh getSolution();
	const TetrahedralMesh* getSource();
	TetrahedralMesh* getTarget();

	TriangleMesh getReconstructedImplicitMesh();

	const std::vector<bool>& getSourceFixedVerts();
	const std::vector<bool>& getTargetFixedVerts();

	const Eigen::MatrixXi& getCorrespondences();

	double getE();
	double getv();
	double getDensity();

	bool getGravityOn();
};


#endif
