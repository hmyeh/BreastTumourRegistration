#include "experiments.h"

#include <iostream>
// Below header is c++17 specific
#include <filesystem>

#include <igl/readCSV.h>

#include "util.h"
#include "evaluationmetric.h"

ExperimentSuite::ExperimentSuite(std::string data_folder, std::string output_folder) : output_folder(output_folder) {
	// Create directory with C++17 filesystem header if not exists
	std::filesystem::create_directory(output_folder);

	// read source mesh
	this->source = TetrahedralMesh::readTriangleMesh(data_folder, source_file);
	this->fixed_verts = readFixedVerticesFile(source.getNumVertices(), data_folder + fixed_verts_file);
	igl::readCSV(data_folder + correspondences_file, correspondences);

	// experiment markers
	igl::readCSV(data_folder + O1_4_markers_file, O1_4);
	igl::readCSV(data_folder + Oa_d_markers_file, Oa_d);
	igl::readCSV(data_folder + I1_4_markers_file, I1_4);

	this->generate_configs_scene = Scene(this->source, this->fixed_verts, num_samp_subspace, true, gravity, E_list[0], v, density, penalty_weight);

	// Read the implicit tumour files and compute the barycentric coordinates
	for (std::string tumour_loc : tumour_locations) {
		// Read the implicit triangle mesh
		TriangleMesh implicit_mesh = TriangleMesh::readFile(data_folder, tumour_loc + "_" + this->source_file);
		// Store the barycentric mappings
		this->tumour_maps[tumour_loc] = ImplicitGeometry(this->generate_configs_scene.getSource(), implicit_mesh);
	}

}


// boolean vector according to the defined experiment_names
void ExperimentSuite::run(std::vector<bool> experiments) {
	// Create directory for each experiment that should be performed
	for (int i = 0; i < experiments.size(); ++i) {
		if (experiments[i]) {
			std::string experiment_name = experiment_type_str[i];
			// Create directory with C++17 filesystem header if not exists
			std::filesystem::create_directory(output_folder + experiment_name + "/");
		}
	}

	// Run the experiments with each value of E 
	for (double E_config : this->E_list) {
		std::string current_output_dir = this->output_folder + "E" + std::to_string(static_cast<int>(E_config)) + "/";
		// Create directory with C++17 filesystem header if not exists
		std::filesystem::create_directory(current_output_dir);

		// Change value of E for material model
		this->generate_configs_scene.changeMaterialParams(E_config, v, density);

		// The supine mesh for deformation
		this->generate_configs_scene.changeGravitationalDirection(this->gravity);
		this->generate_configs_scene.simulate(deformation_params);
		TetrahedralMesh supine_mesh = this->generate_configs_scene.getSolution();
		supine_mesh.writeToFile(current_output_dir, "supine");

		// The prone mesh for deformation
		this->generate_configs_scene.reset();
		this->generate_configs_scene.changeGravitationalDirection(-this->gravity);
		this->generate_configs_scene.simulate(deformation_params);
		TetrahedralMesh prone_mesh = this->generate_configs_scene.getSolution();
		prone_mesh.writeToFile(current_output_dir, "prone");

		// evaluate initial tumour distances
		for (auto const& [tumour_loc, tumour] : this->tumour_maps) {
			std::string tumour_output_dir = current_output_dir + tumour_loc + "/";
			std::filesystem::create_directory(tumour_output_dir);

			TriangleMesh prone_tumour = tumour.reconstructMesh(prone_mesh.getVertices());
			TriangleMesh supine_tumour = tumour.reconstructMesh(supine_mesh.getVertices());
			EvaluationMetric metrics(&prone_tumour, &supine_tumour);
			std::cout << "Evaluation metrics Prone - Supine" << std::endl;
			std::cout << "Centroid euclidean distance in m: " << metrics.computeCentroidDistance() << std::endl;
			metrics.writeScoresToFile(tumour_output_dir, "scores.csv");
			// write the meshes as well
			prone_tumour.writeFile(tumour_output_dir, "implicit_prone");
			supine_tumour.writeFile(tumour_output_dir, "implicit_supine");
		}

		// Create the two required scenes
		Scene prone_scene(prone_mesh, supine_mesh, this->fixed_verts, this->fixed_verts, this->correspondences, num_samp_subspace, true, gravity, E, v, density, penalty_weight);
		Scene rest_shape_scene(this->source, supine_mesh, this->fixed_verts, this->fixed_verts, this->correspondences, num_samp_subspace, true, gravity, E, v, density, penalty_weight);

		// Run the experiments
		for (int i = 0; i < experiments.size(); ++i) {
			// Skip experiment if not wanted
			if (!experiments[i]) {
				continue;
			}

			std::cout << experiment_type_str[i] << " E " << std::to_string(E_config) << std::endl;
			std::string experiment_output_dir = this->output_folder + experiment_type_str[i] + "/E" + std::to_string(static_cast<int>(E_config)) + "/";
			std::filesystem::create_directory(experiment_output_dir);

			switch (static_cast<ExperimentType>(i)) {
			case ExperimentType::RESTSHAPE:
				this->runRestShapeExperiment(&prone_scene, &rest_shape_scene, experiment_output_dir);
				break;
			case ExperimentType::ROTATEDGRAVITY:
				this->runRotatedGravityExperiment(&prone_scene, experiment_output_dir);
				break;
			case ExperimentType::MISALIGNMENT:
				this->runMisalignmentExperiment(&prone_scene, experiment_output_dir);
				break;
			case ExperimentType::UNKNOWNMATERIAL:
				this->runUnknownMaterialExperiment(&prone_scene, experiment_output_dir);
				break;
			case ExperimentType::CORRESPONDENCEWEIGHTTUNING:
				this->runCorrWeightTuningExperiment(&prone_scene, experiment_output_dir);
				break;
			case ExperimentType::PRACTICALSETTINGS:
				this->runPracticalSettingsExperiment(&prone_scene, experiment_output_dir);
				break;
			case ExperimentType::MARKERS:
				this->runMarkerExperiment(&prone_scene, experiment_output_dir);
				break;
			default:
				std::cout << "Experiment Type does not exist" << std::endl;
			}
		}

	}
}


void ExperimentSuite::runMarkerExperiment(Scene* prone_scene, std::string output_dir) {
	Eigen::MatrixXi all_markers(O1_4.rows() + Oa_d.rows() + I1_4.rows(), O1_4.cols());
	all_markers << O1_4, Oa_d, I1_4;
	Eigen::MatrixXi combi_1(O1_4.rows() + Oa_d.rows(), O1_4.cols());
	combi_1 << O1_4, Oa_d;
	Eigen::MatrixXi combi_2(O1_4.rows() + I1_4.rows(), O1_4.cols());
	combi_2 << O1_4, I1_4;
	Eigen::MatrixXi combi_3(Oa_d.rows() + I1_4.rows(), Oa_d.cols());
	combi_3 << Oa_d, I1_4;

	prone_scene->reset();
	Solver::Parameters full_params = full_method_params;
	std::string current_output_dir = output_dir + "all/";
	full_params.output_path = current_output_dir;
	std::filesystem::create_directory(full_params.output_path);
	prone_scene->setCorrespondences(all_markers);
	prone_scene->simulate(full_params);
	prone_scene->evaluate(this->tumour_maps, current_output_dir);

	prone_scene->reset();
	current_output_dir = output_dir + "O1_4Oa_d/";
	full_params.output_path = current_output_dir;
	std::filesystem::create_directory(full_params.output_path);
	prone_scene->setCorrespondences(combi_1);
	prone_scene->simulate(full_params);
	prone_scene->evaluate(this->tumour_maps, current_output_dir);

	prone_scene->reset();
	current_output_dir = output_dir + "O1_4I1_4/";
	full_params.output_path = current_output_dir;
	std::filesystem::create_directory(full_params.output_path);
	prone_scene->setCorrespondences(combi_2);
	prone_scene->simulate(full_params);
	prone_scene->evaluate(this->tumour_maps, current_output_dir);

	prone_scene->reset();
	current_output_dir = output_dir + "Oa_dI1_4/";
	full_params.output_path = current_output_dir;
	std::filesystem::create_directory(full_params.output_path);
	prone_scene->setCorrespondences(combi_3);
	prone_scene->simulate(full_params);
	prone_scene->evaluate(this->tumour_maps, current_output_dir);
}

// Run the experiment for each of the three practical settings (landmarks only, surface scan only, both)
void ExperimentSuite::runPracticalSettingsExperiment(Scene* prone_scene, std::string output_dir) {
	prone_scene->reset();
	Solver::Parameters landmark_params = landmark_deformation_params;
	landmark_params.output_path = output_dir + "landmarks/";
	std::filesystem::create_directory(landmark_params.output_path);
	prone_scene->simulate(landmark_params);
	prone_scene->evaluate(this->tumour_maps, landmark_params.output_path);

	prone_scene->reset();
	Solver::Parameters surface_params = surface_deformation_params;
	surface_params.output_path = output_dir + "surface/";
	std::filesystem::create_directory(surface_params.output_path);
	prone_scene->simulate(surface_params);
	prone_scene->evaluate(this->tumour_maps, surface_params.output_path);

	prone_scene->reset();
	Solver::Parameters full_params = full_method_params;
	full_params.output_path = output_dir + "full_method/";
	std::filesystem::create_directory(full_params.output_path);
	prone_scene->simulate(full_params);
	prone_scene->evaluate(this->tumour_maps, full_params.output_path);

	prone_scene->reset();
	Solver::Parameters no_landmarks_params = full_method_no_landmarks_params;
	no_landmarks_params.output_path = output_dir + "no_landmarks/";
	std::filesystem::create_directory(no_landmarks_params.output_path);
	prone_scene->simulate(no_landmarks_params);
	prone_scene->evaluate(this->tumour_maps, no_landmarks_params.output_path);
}


void ExperimentSuite::runCorrWeightTuningExperiment(Scene* prone_scene, std::string output_dir) {

	for (double corr_weight : corr_weight_list) {
		std::string current_output_dir = output_dir + std::to_string(corr_weight) + "/";
		std::filesystem::create_directory(current_output_dir);

		prone_scene->reset();
		Solver::Parameters full_params = full_method_params;
		full_params.output_path = current_output_dir;
		full_params.landmark_weight = corr_weight;
		prone_scene->simulate(full_params);
		prone_scene->evaluate(this->tumour_maps, current_output_dir);
	}

}

void ExperimentSuite::runRestShapeExperiment(Scene* prone_scene, Scene* rest_shape_scene, std::string output_dir) {
	// Prone experiment with the full method
	prone_scene->reset();
	Solver::Parameters full_params = full_method_params;
	full_params.output_path = output_dir + "prone/";
	std::filesystem::create_directory(full_params.output_path);
	prone_scene->simulate(full_params);
	prone_scene->evaluate(this->tumour_maps, full_params.output_path);

	// Rest shape experiment with the full method
	rest_shape_scene->reset();
	full_params.output_path = output_dir + "rest/";
	std::filesystem::create_directory(full_params.output_path);
	rest_shape_scene->simulate(full_params);
	rest_shape_scene->evaluate(this->tumour_maps, full_params.output_path);
}

void ExperimentSuite::runRotatedGravityExperiment(Scene* prone_scene, std::string output_dir) {
	// Ignoring  i = 0 for rotated gravity because that does not change the direction of gravity
	for (int i = 2; i < 3; ++i) {
		// Create directory for rotation axis
		std::string current_axis_dir = output_dir + std::to_string(i) + "/";
		std::filesystem::create_directory(current_axis_dir);

		// Set rotation axis
		Eigen::Vector3d rotation_axis = Eigen::Vector3d::Zero();
		rotation_axis(i) = 1.0;

		for (double rotation : this->rotations_list) {
			std::string current_output_dir = current_axis_dir + std::to_string(static_cast<int>(rotation)) + "/";
			std::filesystem::create_directory(current_output_dir);
			Eigen::Matrix3d rot_mat = getRotationMatrix(rotation, rotation_axis);
			//prone_energy_function->setGravityVector(rot_mat * this->gravity);

			prone_scene->reset();
			prone_scene->changeGravitationalDirection(rot_mat * this->gravity);
			Solver::Parameters full_params = full_method_params;
			full_params.output_path = current_output_dir;
			prone_scene->simulate(full_params);
			prone_scene->evaluate(this->tumour_maps, current_output_dir);
		}
	}
}


void ExperimentSuite::runMisalignmentExperiment(Scene* prone_scene, std::string output_dir) {
	std::filesystem::create_directory(output_dir + "rotation/");

	for (int i = 0; i < 2; ++i) {
		// Create directory for rotation axis
		std::string current_axis_dir = output_dir + "rotation/" + std::to_string(i) + "/";
		std::filesystem::create_directory(current_axis_dir);

		// Set rotation axis
		Eigen::Vector3d rotation_axis = Eigen::Vector3d::Zero();
		rotation_axis(i) = 1.0;

		for (double rotation : this->rotations_list) {
			std::string current_output_dir = current_axis_dir + std::to_string(static_cast<int>(rotation)) + "/";
			std::filesystem::create_directory(current_output_dir);
			Eigen::Matrix3d rot_mat = getRotationMatrix(rotation, rotation_axis);
			std::cout << "Current misalignment rotation: " << std::endl << rot_mat << std::endl;

			prone_scene->reset();
			prone_scene->getTarget()->applyRotation(rot_mat);
			prone_scene->getTarget()->writeToFile(current_output_dir, "target");
			Solver::Parameters full_params = full_method_params;
			full_params.output_path = current_output_dir;
			prone_scene->simulate(full_params);
			prone_scene->evaluate(this->tumour_maps, current_output_dir);
			// TODO: counteract rotation should be solved differently
			prone_scene->getTarget()->applyRotation(rot_mat.transpose());
		}
	}

	// Translation
	std::filesystem::create_directory(output_dir + "translation/");


	for (int i = 0; i < 2; ++i) {
		// Create directory for translation axis
		std::string current_axis_dir = output_dir + "translation/" + std::to_string(i) + "/";
		std::filesystem::create_directory(current_axis_dir);

		// Set rotation axis
		Eigen::Vector3d translation_axis = Eigen::Vector3d::Zero();
		translation_axis(i) = 1.0;

		for (double translation : this->translations_list) {
			std::string current_output_dir = current_axis_dir + std::to_string(translation) + "/";
			std::filesystem::create_directory(current_output_dir);
			Eigen::Vector3d transl_vec = translation_axis * translation;
			std::cout << "Current misalignment translation: " << transl_vec.transpose() << std::endl;

			prone_scene->reset();
			prone_scene->getTarget()->applyTranslation(transl_vec);
			prone_scene->getTarget()->writeToFile(current_output_dir, "target");
			Solver::Parameters full_params = full_method_params;
			full_params.output_path = current_output_dir;
			prone_scene->simulate(full_params);
			prone_scene->evaluate(this->tumour_maps, current_output_dir);
			// TODO: counteract translation should be solved differently
			prone_scene->getTarget()->applyTranslation(-transl_vec);

		}
	}

}

void ExperimentSuite::runUnknownMaterialExperiment(Scene* prone_scene, std::string output_dir) {
	// Assumption: single area

	for (double E_experiment : this->E_list) {
		std::string current_output_dir = output_dir + std::to_string(static_cast<int>(E_experiment)) + "/";
		std::filesystem::create_directory(current_output_dir);

		prone_scene->reset();
		prone_scene->changeMaterialParams(E_experiment, v, density);
		Solver::Parameters full_params = full_method_params;
		full_params.output_path = current_output_dir;
		prone_scene->simulate(full_params);
		prone_scene->evaluate(this->tumour_maps, current_output_dir);
	}

	// Reset material params
	prone_scene->changeMaterialParams(E, v, density);

}
