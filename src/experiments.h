#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H

#include <string>
#include <vector>
#include <map>
#include <string>

#include <Eigen/Dense>

#include "tetrahedralmesh.h"
#include "scene.h"


enum class ExperimentType {
	RESTSHAPE,
	ROTATEDGRAVITY,
	MISALIGNMENT,
	UNKNOWNMATERIAL,
	CORRESPONDENCEWEIGHTTUNING,
	PRACTICALSETTINGS,
	MARKERS
};
static const std::vector<std::string> experiment_type_str { "RestShapeExperiment", "RotatedGravityExperiment", "MisalignmentExperiment", "UnknownMaterialExperiment", "CorrespondenceWeightTuningExperiment", "PracticalSettingsExperiment", "MarkerExperiment"};



//bool simulation, bool non_rigid_alignment, bool reduced_sim, bool use_landmarks, bool debug_mode
const Solver::Parameters deformation_params(true, false, false, false, false);
const Solver::Parameters landmark_deformation_params(true, false, true, true, false);
const Solver::Parameters surface_deformation_params(false, true, false, false, false);
const Solver::Parameters full_method_params(true, true, true, true, false);
const Solver::Parameters full_method_no_landmarks_params(true, true, true, false, false);



// Experiment suite 
// All of the virtual experiments should be called from here
class ExperimentSuite {
protected:
	std::string output_folder;

	// The meshes
	TetrahedralMesh source;
	//std::unique_ptr<TetrahedralMesh> target;

	std::string source_file = "rest";
	std::string fixed_verts_file = "fixed_verts.txt";
	std::string correspondences_file = "correspondences.csv";

	Scene generate_configs_scene;

	// Energy function based on the source with correspondence constraints with the target
	EnergyFunction energy_function;

	// vector containing all the fixed vertices
	std::vector<bool> fixed_verts;

	// Linear skinning subspace
	Eigen::MatrixXd subspace;

	// Correspondences between source and target
	Eigen::MatrixXi correspondences;

	// test markers
	Eigen::MatrixXi O1_4;
	Eigen::MatrixXi Oa_d;
	Eigen::MatrixXi I1_4;
	std::string O1_4_markers_file = "O1_4.csv";
	std::string Oa_d_markers_file = "Oa_d.csv";
	std::string I1_4_markers_file = "I1_4.csv";


	// List of material parameter Young's modulus E
	std::vector<double> E_list{  200.0, 400.0, 600.0, 800.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0 };//  Commented since bad results passing itself->50.0, 100.0,

	double E = 3400.0;
	// Material parameter Poisson's ratio v
	double v = 0.49;
	// Density rho
	double density = 1000.0;

	// List of correspondence weights to test
	std::vector<double> corr_weight_list{1.0, 10.0, 100.0, 1000.0};

	// List of tumour locations
	std::vector<std::string> tumour_locations { "middle", "top", "left", "bottom", "right" };
	//std::vector<ImplicitGeometry> tumour_barycentric_maps;
	std::map<std::string, ImplicitGeometry> tumour_maps;

	// List of rotations for the experiments (both misalignment rotation + rotated gravity)
	std::vector<double> rotations_list { 2, 4, 6, 8, 10, 15 };

	// List of translations for the misalignment translation experiment
	std::vector<double> translations_list{ -0.01, -0.008, -0.006, -0.004, -0.002, 0.002, 0.004, 0.006, 0.008, 0.01};

	// Number of samples for subspace
	int num_samp_subspace = 50;
	// Gravity direction down
	Eigen::Vector3d gravity = Eigen::Vector3d(9.81, 0, 0);
	// Global soft penalty weight
	double penalty_weight = 100.0;


	// Solver constants
	OptimizationMethods optimization_method = OptimizationMethods::NEWTON;
	int max_iterations = 100;


public:

	// The one setting up the experiment suite should know the setup params specific for the mesh
	ExperimentSuite(std::string data_folder, std::string output_folder);

	// boolean vector according to the defined experiment_names
	void run(std::vector<bool> experiments);

	void runMarkerExperiment(Scene* prone_scene, std::string output_dir);
	// Run the experiment for each of the three practical settings (landmarks only, surface scan only, both)
	void runPracticalSettingsExperiment(Scene* prone_scene, std::string output_dir);
	void runCorrWeightTuningExperiment(Scene* prone_scene, std::string output_dir);
	void runRestShapeExperiment(Scene* prone_scene, Scene* rest_shape_scene, std::string output_dir);
	void runRotatedGravityExperiment(Scene* prone_scene, std::string output_dir);
	void runMisalignmentExperiment(Scene* prone_scene, std::string output_dir);
	void runUnknownMaterialExperiment(Scene* prone_scene, std::string output_dir);

};


#endif
