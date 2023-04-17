#include <algorithm>

#include "viewer.h"
#include "util.h"
#include "experiments.h"

Viewer::Viewer() : num_meshes(0) {
	this->viewer = igl::opengl::glfw::Viewer();

	// Set for multiple viewports (https://github.com/libigl/libigl/blob/main/tutorial/108_MultipleViews/main.cpp)
	viewer.core().viewport = Eigen::Vector4f(0, 0, 1260, 720);
	this->left_view = viewer.core_list[0].id;
	this->middle_view = viewer.append_core(Eigen::Vector4f(420, 0, 1260, 720));
	this->right_view = viewer.append_core(Eigen::Vector4f(840, 0, 1260, 720));

	viewer.callback_post_resize = [&](igl::opengl::glfw::Viewer& v, int w, int h) {
		v.core(this->left_view).viewport = Eigen::Vector4f(0, 0, w / 3, h);
		v.core(this->middle_view).viewport = Eigen::Vector4f(w / 3, 0,  w / 3, h);
		v.core(this->right_view).viewport = Eigen::Vector4f(w / 3 * 2, 0, w / 3, h);
		return true;
	};

	// set default mesh invisible
	this->viewer.data().set_visible(false, this->left_view);
	this->viewer.data().set_visible(false, this->middle_view);
	this->viewer.data().set_visible(false, this->right_view);
}

void Viewer::setScene(Scene* scene) {
	this->scene = scene;

	// TODO: cleanup previous meshes
	// correspondence fixed verts (this is for the icohalfsphere test case)
	std::vector<bool> source_landmarks(static_cast<unsigned long>(this->scene->getSource()->getNumVertices()), false);
	std::vector<bool> target_landmarks(static_cast<unsigned long>(this->scene->getTarget()->getNumVertices()), false);
	for (int i = 0; i < this->scene->getCorrespondences().rows(); ++i) {
		source_landmarks[this->scene->getCorrespondences()(i, 0)] = true;
		target_landmarks[this->scene->getCorrespondences()(i, 1)] = true;
	}

	// Also set everything ready to visualize the new scene
	this->source_mesh_id = this->appendMesh(this->left_view, this->scene->getSource()->getVertices(), this->scene->getSource()->getTriangles());
	// Append the deformed pose for the solution
	this->solution_mesh_id = this->appendMesh(this->middle_view, this->scene->getSolution().getVertices(), this->scene->getSolution().getTriangles(), source_landmarks);
	
	TriangleMesh tumour = this->scene->getReconstructedImplicitMesh();
	Eigen::MatrixXd tumour_verts = tumour.getVertices();
	Eigen::MatrixXi tumour_tris = tumour.getTriangles();
	this->tumour_mesh_id = this->appendMesh(this->middle_view, tumour_verts, tumour_tris);
	//int solution_mesh_id = viewer.appendMesh(viewer.middle_view, test_rest_view, solution_mesh->getTriangles(), input_fixed_verts);
	// Append the deformed pose
	this->target_mesh_id = this->appendMesh(this->right_view, this->scene->getTarget()->getVertices(), this->scene->getTarget()->getTriangles(), target_landmarks);
	this->drawViewerMenu();
}

// Returns data_list_id for appended mesh to viewer
int Viewer::appendMesh(unsigned int viewport_id, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
	int mesh_id = 0;
	if (this->num_meshes != 0) {
		// Append a new mesh and set it to invisible
		mesh_id = this->viewer.append_mesh(false);
	}

	// Set the mesh and also visible to the specified viewport
	this->setMesh(viewport_id, mesh_id, V, F);

	this->num_meshes++;
	return mesh_id;
}

int Viewer::appendMesh(unsigned int viewport_id, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::vector<bool> fixedVerts) {
	int mesh_id = this->appendMesh(viewport_id, V, F);
	this->setOverlayPoints(mesh_id, V, fixedVerts);
	return mesh_id;
}

void Viewer::setAxes(unsigned int viewport_id, int mesh_id) {
	// Set axis
	Eigen::MatrixXd axes1(3, 3);
	axes1 << 0, 0, 0,
		0, 0, 0,
		0, 0, 0;
	Eigen::MatrixXd axes2(3, 3);
	axes2 << 1, 0, 0,
		0, 1, 0,
		0, 0, 1;
	// Use axes2 as the colour input
	this->viewer.data(mesh_id).add_edges(axes1, axes2, axes2);
}

void Viewer::setOverlayPoints(int mesh_id, const Eigen::MatrixXd& points, std::vector<bool> fixedPoints) {
	Eigen::MatrixXd pointsColours(fixedPoints.size(), 3);
	for (unsigned int i = 0; i < fixedPoints.size(); ++i) {
		if (fixedPoints[i]) {
			pointsColours.row(i) = Eigen::Vector3d(1, 0, 0); // Red for fixed points
		}
		else {
			pointsColours.row(i) = Eigen::Vector3d(0, 0, 0); // Black for all other points
		}
	}

	// setting fixed points overlay with points
	this->viewer.data(mesh_id).add_points(points, pointsColours);
	this->viewer.data(mesh_id).point_size = 5.f;
}

void Viewer::setMesh(unsigned int viewport_id, int mesh_id, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
	// Clear the mesh at mesh_id 
	this->viewer.data(mesh_id).clear();
	this->viewer.data(mesh_id).set_mesh(V, F);

	// Set axis
	this->setAxes(viewport_id, mesh_id);

	// Set mesh visible to specific viewport
	this->viewer.data(mesh_id).set_visible(true, viewport_id);
	this->viewer.core(viewport_id).align_camera_center(V, F);
}

void Viewer::setMesh(unsigned int viewport_id, int mesh_id, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::vector<bool> fixedVerts) {
	this->setMesh(viewport_id, mesh_id, V, F);
	this->setOverlayPoints(mesh_id, V, fixedVerts);
}

void Viewer::deleteMesh(int mesh_id) {
	viewer.erase_mesh(viewer.mesh_index(mesh_id));
}

void Viewer::alignAllViewerCameras(Eigen::MatrixXd& V, Eigen::MatrixXi& F) {
	int mesh_id = this->viewer.append_mesh(true);
	this->viewer.data(mesh_id).clear();
	this->viewer.data(mesh_id).set_mesh(V, F);

	this->viewer.core(this->left_view).align_camera_center(V, F);
	this->viewer.core(this->middle_view).align_camera_center(V, F);
	this->viewer.core(this->right_view).align_camera_center(V, F);

	this->viewer.data(mesh_id).set_visible(false, this->left_view);
	this->viewer.data(mesh_id).set_visible(false, this->middle_view);
	this->viewer.data(mesh_id).set_visible(false, this->right_view);


	this->viewer.core(this->left_view).camera_zoom = 1.0;
	this->viewer.core(this->middle_view).camera_zoom = 1.0;
	this->viewer.core(this->right_view).camera_zoom = 1.0;
}

void Viewer::launch() {
	this->viewer.launch();
}

void setVibrationModeInViewer(int mode, Viewer* viewer, unsigned int input_mesh_id, const Eigen::MatrixXd& eigenvalues, const Eigen::MatrixXd& eigenvectors, const Eigen::MatrixXd& verts, const Eigen::MatrixXi& faces) {
	Eigen::MatrixXd col = eigenvectors.col(mode);

	Eigen::MatrixXd vibrationMode = verts.transpose();
	Eigen::MatrixXd u = Eigen::Map<Eigen::MatrixXd>(col.data(), vibrationMode.rows(), vibrationMode.cols());
	//std::cout << u << std::endl;
	vibrationMode = vibrationMode + u;
	vibrationMode.transposeInPlace();
	viewer->setMesh(viewer->left_view, input_mesh_id, vibrationMode, faces);
}

void Viewer::drawViewerMenu() {
	// Attach a menu plugin
	plugin = igl::opengl::glfw::imgui::ImGuiPlugin();
	viewer.plugins.push_back(&plugin);
	menu = igl::opengl::glfw::imgui::ImGuiMenu();
	plugin.widgets.push_back(&menu);

	// Add content to the default menu window
	menu.callback_draw_viewer_menu = [this]()
	{
		// Draw parent menu content
		//this->menu.draw_viewer_menu();

		 //The experiments can be run with a button for each experiment group/type
		if (ImGui::CollapsingHeader("Experiments")) {
			static bool rotate_grav_exp = true;
			static bool rest_shape_exp = true;
			static bool misalignment_exp = true;
			static bool unknown_material_exp = true;
			static bool corr_weight_tuning_exp = true;
			static bool practical_settings_exp = true;
			static bool markers_exp = true;

			ImGui::Checkbox("Rotated Gravity Experiments", &rotate_grav_exp);
			ImGui::Checkbox("Rest Shape Experiments", &rest_shape_exp);
			ImGui::Checkbox("Misalignment Experiments", &misalignment_exp);
			ImGui::Checkbox("Unknown Material Experiments", &unknown_material_exp);
			ImGui::Checkbox("Correspondence Weight Tuning Experiments", &corr_weight_tuning_exp);
			ImGui::Checkbox("Practical Settings Experiments", &practical_settings_exp);
			ImGui::Checkbox("Markers Experiments", &markers_exp);

			// Run the selected experiments
			if (ImGui::Button("Run selected experiments")) {
				std::string data_folder = "data/experiments/";
				std::string output_folder = "output/experiments/";
				std::string fixed_verts_file = "fixed_verts.txt";
				std::string correspondence_verts_file = "correspondences.csv";
				ExperimentSuite experiments(data_folder, output_folder);
				std::vector<bool> exp_to_run{ rest_shape_exp, rotate_grav_exp, misalignment_exp, unknown_material_exp, corr_weight_tuning_exp, practical_settings_exp, markers_exp };
				experiments.run(exp_to_run);
			}
		}

		if (ImGui::CollapsingHeader("Energy Function")) {
			static double E = scene->getE();
			static double v = scene->getv();
			static double density = scene->getDensity();
			// Adjust the material model parameters
			if (ImGui::InputDouble("E", &E)) {
				scene->changeMaterialParams(E, v, density);
			}
			if (ImGui::InputDouble("v", &v)) {
				scene->changeMaterialParams(E, v, density);
			}
			if (ImGui::InputDouble("Density", &density)) {
				scene->changeMaterialParams(E, v, density);
			}
			static bool gravity_on = scene->getGravityOn();
			if (ImGui::Checkbox("Gravity On", &gravity_on)) {
				scene->setGravityOn(gravity_on);
			}
		}

		static Solver::Parameters solver_params;
		//solver_params.landmark_weight = 1.0;

		if (ImGui::CollapsingHeader("Solver")) {
			static bool debug_mode = true;
			if (ImGui::Checkbox("Debug Mode", &debug_mode)) {
				solver_params.debug_mode = debug_mode;
			}
			static std::string output_path = "output/";
			if (ImGui::InputText("Output Directory", output_path)) {
				solver_params.output_path = output_path;
			}

			// Simulation options
			static OptimizationMethods optimization_method = OptimizationMethods::NEWTON;
			ImGui::Combo("Optimization Method", (int*)(&optimization_method), "GRADIENT_DESCENT\0NEWTON\0\0");

			if (ImGui::CollapsingHeader("Landmark Guided Deformation")) {
				ImGui::InputInt("Max Iterations", &solver_params.simulation_max_iter);
				ImGui::Checkbox("Reduced Simulation", &solver_params.reduced_sim);
				ImGui::Checkbox("Use Landmarks", &solver_params.use_landmarks);
				if (ImGui::Button("Landmark Constrained Solve")) {
					solver_params.simulation = true;
					solver_params.non_rigid_alignment = false;
					scene->simulate(solver_params);
					// Update mesh in view
					this->setMesh(this->middle_view, this->solution_mesh_id, scene->getSolution().getVertices(), scene->getSolution().getTriangles(), scene->getSourceFixedVerts());
					TriangleMesh tumour = this->scene->getReconstructedImplicitMesh();
					Eigen::MatrixXd tumour_verts = tumour.getVertices();
					Eigen::MatrixXi tumour_tris = tumour.getTriangles();
					this->setMesh(this->middle_view, this->tumour_mesh_id, tumour_verts, tumour_tris);
				}
			}

			if (ImGui::CollapsingHeader("Surface Guided Deformation")) {
				ImGui::InputInt("Max Iterations", &solver_params.non_rigid_alignment_max_iter);
				if (ImGui::Button("Surface Projection Solve")) {
					solver_params.simulation = false;
					solver_params.non_rigid_alignment = true;
					scene->simulate(solver_params);
					// Update mesh in view
					this->setMesh(this->middle_view, this->solution_mesh_id, scene->getSolution().getVertices(), scene->getSolution().getTriangles(), scene->getSourceFixedVerts());
					TriangleMesh tumour = this->scene->getReconstructedImplicitMesh();
					Eigen::MatrixXd tumour_verts = tumour.getVertices();
					Eigen::MatrixXi tumour_tris = tumour.getTriangles();
					this->setMesh(this->middle_view, this->tumour_mesh_id, tumour_verts, tumour_tris);
				}
			}

			if (ImGui::CollapsingHeader("Full Method")) {
				if (ImGui::Button("Solve")) {
					solver_params.simulation = true;
					solver_params.non_rigid_alignment = true;
					scene->simulate(solver_params);
					// Update mesh in view
					this->setMesh(this->middle_view, solution_mesh_id, scene->getSolution().getVertices(), scene->getSolution().getTriangles(), scene->getSourceFixedVerts());
					TriangleMesh tumour = this->scene->getReconstructedImplicitMesh();
					Eigen::MatrixXd tumour_verts = tumour.getVertices();
					Eigen::MatrixXi tumour_tris = tumour.getTriangles();
					this->setMesh(this->middle_view, this->tumour_mesh_id, tumour_verts, tumour_tris);
				}
			}
		}

		if (ImGui::CollapsingHeader("Evaluation")) {
			if (ImGui::Button("Run evaluation measures")) {
				scene->evaluate();
			}
		}

	};
}
