#include "scene.h"

#include <iostream>
// Below header is c++17 specific
#include <filesystem>

#include <igl/readCSV.h>

#include "materialmodel.h"
#include "constraint.h"
#include "meshsampler.h"
#include "evaluationmetric.h"
#include "util.h"


Scene::Scene(const TetrahedralMesh& source, const std::vector<bool>& source_fixed_verts, int num_sample_verts_subspace, bool gravity_on, Eigen::Vector3d gravity, double E, double v, double density, double penalty_weight) :
	source(source), solution(TetrahedralMesh(source)), num_samples_subspace(num_sample_verts_subspace), source_fixed_verts(source_fixed_verts), E(E), v(v), density(density) {
	// Construct subspace
	std::vector<int> source_landmarks;
	source_landmarks.resize(correspondences.rows());
	Eigen::Map<Eigen::MatrixXi>(source_landmarks.data(), correspondences.rows(), 1) = correspondences.col(0);

	MeshSampler mesh_sampler(solution.getVertices(), solution.getTetrahedrons(), source_landmarks);
	this->subspace = mesh_sampler.createSkinningSpace(num_sample_verts_subspace);

	// Material Parameters for the areas
	double mu;
	double lambda;
	convertToLameParameters(E, v, mu, lambda);
	std::unique_ptr<MaterialModel> material_model(new StableNeoHookeanModel(mu, lambda));
	std::vector<std::unique_ptr<Element::Parameters> > area_params;
	area_params.push_back(std::make_unique<Element::Parameters>(std::move(material_model), density));

	this->energy_function = EnergyFunction(&solution, std::move(area_params), penalty_weight, gravity_on, gravity);
}

Scene::Scene(const TetrahedralMesh& source, const TetrahedralMesh& target, const std::vector<bool>& source_fixed_verts, const std::vector<bool>& target_fixed_verts, const Eigen::MatrixXi& correspondences,
	int num_sample_verts_subspace, bool gravity_on, Eigen::Vector3d gravity, double E, double v, double density, double penalty_weight) :
	source(source), target(target), solution(TetrahedralMesh(source)), source_fixed_verts(source_fixed_verts), target_fixed_verts(target_fixed_verts), num_samples_subspace(num_sample_verts_subspace),
	correspondences(correspondences), E(E), v(v), density(density) {
	// Construct subspace
	std::vector<int> source_landmarks;
	source_landmarks.resize(correspondences.rows());
	Eigen::Map<Eigen::MatrixXi>(source_landmarks.data(), correspondences.rows(), 1) = correspondences.col(0);

	MeshSampler mesh_sampler(solution.getVertices(), solution.getTetrahedrons(), source_landmarks);
	this->subspace = mesh_sampler.createSkinningSpace(num_sample_verts_subspace);

	// Material Parameters for the areas
	double mu;
	double lambda;
	convertToLameParameters(E, v, mu, lambda);
	std::unique_ptr<MaterialModel> material_model(new StableNeoHookeanModel(mu, lambda));
	std::vector<std::unique_ptr<Element::Parameters> > area_params;
	area_params.push_back(std::make_unique<Element::Parameters>(std::move(material_model), density));

	this->energy_function = EnergyFunction(&solution, std::move(area_params), penalty_weight, gravity_on, gravity);

}


Scene::Scene(std::string data_folder, std::string source_file, std::string target_file, std::string source_fixed_verts_file, std::string target_fixed_verts_file, std::string source_tumour_file, std::string target_tumour_file, std::string correspondences_file,
	bool rigid_alignment, int num_sample_verts_subspace, bool gravity_on, Eigen::Vector3d gravity, double E, double v, double density, double penalty_weight) : num_samples_subspace(num_sample_verts_subspace), E(E), v(v), density(density) {
	// Create the tetrahedral mesh and implicit barycentric mapping
	this->source = TetrahedralMesh::readTriangleMesh(data_folder, source_file);
	this->target = TetrahedralMesh::readTriangleMesh(data_folder, target_file);

	// Read the files containing the fixed points
	this->source_fixed_verts = readFixedVerticesFile(source.getNumVertices(), data_folder + source_fixed_verts_file);
	this->target_fixed_verts = readFixedVerticesFile(target.getNumVertices(), data_folder + target_fixed_verts_file);

	igl::readCSV(data_folder + correspondences_file, correspondences);

	// implicit geometry
	this->implicit_solution = ImplicitGeometry(&source, TriangleMesh::readFile(data_folder, source_tumour_file));
	this->implicit_target = TriangleMesh::readFile(data_folder, target_tumour_file);

	// Automatic rigid alignment (ICP) if needed fix for separate impl geom TODO
	if (rigid_alignment) {
		TetrahedralMesh::rigidAlignment(&source, source_fixed_verts, &target, target_fixed_verts, &implicit_solution, &implicit_target);
	}

	// A copy for solution
	solution = TetrahedralMesh(source);

	// Construct subspace
	std::vector<int> source_landmarks;
	source_landmarks.resize(correspondences.rows());
	Eigen::Map<Eigen::MatrixXi>(source_landmarks.data(), correspondences.rows(), 1) = correspondences.col(0);

	MeshSampler mesh_sampler(solution.getVertices(), solution.getTetrahedrons(), source_landmarks);
	this->subspace = mesh_sampler.createSkinningSpace(num_sample_verts_subspace);

	// Material Parameters for the areas
	double mu;
	double lambda;
	convertToLameParameters(E, v, mu, lambda);
	std::unique_ptr<MaterialModel> material_model(new StableNeoHookeanModel(mu, lambda));
	std::vector<std::unique_ptr<Element::Parameters> > area_params;
	area_params.push_back(std::make_unique<Element::Parameters>(std::move(material_model), density));

	this->energy_function = EnergyFunction(&solution, std::move(area_params), penalty_weight, gravity_on, gravity);
}


void Scene::simulate(Solver::Parameters solver_params, std::string file_name) {
	Solver::solve(&solution, &target, source_fixed_verts, target_fixed_verts, correspondences, &energy_function, subspace, solver_params);
	// write meshes to file
	this->solution.writeToFile(solver_params.output_path, file_name);
}


void Scene::evaluate(const std::map<std::string, ImplicitGeometry>& tumour_maps, std::string output_path) {
	for (auto const& [tumour_loc, tumour] : tumour_maps) {
		std::string current_output_dir = output_path + tumour_loc + "/";
		// Create directory with C++17 filesystem header if not exists
		std::filesystem::create_directory(current_output_dir);

		this->evaluate(tumour, current_output_dir);
	}
}
void Scene::evaluate(const ImplicitGeometry& implicit_geometry, std::string output_path, std::string file_name) {
	TriangleMesh result = implicit_geometry.reconstructMesh(this->solution.getVertices());
	TriangleMesh ground_truth = implicit_geometry.reconstructMesh(this->target.getVertices());
	EvaluationMetric metrics(&result, &ground_truth);
	std::cout << "Evaluation metrics" << std::endl;
	std::cout << "Centroid euclidean distance in m: " << metrics.computeCentroidDistance() << std::endl;
	metrics.writeScoresToFile(output_path, file_name);
}


void Scene::evaluate(std::string output_path, std::string file_name) {
	TriangleMesh result = implicit_solution.reconstructMesh(solution.getVertices());
	EvaluationMetric metrics(&result, &implicit_target);
	std::cout << "Evaluation metrics" << std::endl;
	std::cout << "Centroid euclidean distance in m: " << metrics.computeCentroidDistance() << std::endl;
	result.writeFile(output_path, file_name);
}


void Scene::changeMaterialParams(double E, double v, double density) {
	this->E = E;
	this->v = v;
	this->density = density;

	double mu;
	double lambda;
	convertToLameParameters(E, v, mu, lambda);
	const std::vector<std::unique_ptr<Element::Parameters> >& area_params = this->energy_function.getMaterialParams();
	area_params[0]->material_model.reset();
	area_params[0]->material_model = std::unique_ptr<MaterialModel>(new StableNeoHookeanModel(mu, lambda));
	area_params[0]->density = density;
}


void Scene::changeGravitationalDirection(Eigen::Vector3d gravity) {
	this->energy_function.setGravityVector(gravity);
}

void Scene::setGravityOn(bool gravity_on) {
	this->energy_function.setGravityOn(gravity_on);
}

void Scene::setCorrespondences(const Eigen::MatrixXi& correspondences) {
	// Construct subspace
	std::vector<int> source_landmarks;
	source_landmarks.resize(correspondences.rows());
	Eigen::Map<Eigen::MatrixXi>(source_landmarks.data(), correspondences.rows(), 1) = correspondences.col(0);

	MeshSampler mesh_sampler(solution.getVertices(), solution.getTetrahedrons(), source_landmarks);
	this->subspace = mesh_sampler.createSkinningSpace(num_samples_subspace);
	this->correspondences = correspondences;
}

void Scene::reset() {
	// Reset solution
	this->solution = TetrahedralMesh(this->source);
}


TetrahedralMesh Scene::getSolution() {
	return this->solution;
}

const TetrahedralMesh* Scene::getSource() {
	return &this->source;
}

TetrahedralMesh* Scene::getTarget() {
	return &this->target;
}

TriangleMesh Scene::getReconstructedImplicitMesh() {
	return this->implicit_solution.reconstructMesh(this->solution.getVertices());
}

const std::vector<bool>& Scene::getSourceFixedVerts() {
	return this->source_fixed_verts;
}

const std::vector<bool>& Scene::getTargetFixedVerts() {
	return this->target_fixed_verts;
}

const Eigen::MatrixXi& Scene::getCorrespondences() {
	return this->correspondences;
}

double Scene::getE() {
	return this->E;
}

double Scene::getv() {
	return this->v;
}

double Scene::getDensity() {
	return this->density;
}

bool Scene::getGravityOn() {
	return this->energy_function.getGravityOn();
}
