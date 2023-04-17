#define EIGEN_USE_MKL_ALL
#define MKL_LP64

#include <string>
#include <iostream>
#include <Eigen/Dense>

#include "viewer.h"
#include "scene.h"


int main(int argc, char * argv[])
{

	try {
		std::string data_folder = "data/experiments/";
		// Path to input mesh and fixed points
		std::string input_mesh_path = "prone";
		std::string fixed_points_input_mesh_path = "fixed_verts.txt";
		// Path to the implicit mesh
		std::string implicit_input_mesh_path = "implicit_prone";

		// Path to input mesh and fixed points
		std::string target_mesh_path = "supine";
		std::string fixed_points_target_mesh_path = "fixed_verts.txt";
		// Path to the implicit mesh
		std::string implicit_target_mesh_path = "implicit_supine";

		// File with correspondences
		std::string correspondences_path = "correspondences.csv";

		// Folder where to write the debug files to
		std::string output_path = "output/";

		int num_sample_verts_subspace = 50; // The number of degrees of freedom for the mesh vertex positions will be 12 times that
		Eigen::Vector3d gravity(9.81, 0, 0); //icosphere
		
		//Youngs modulus with unit Pa = N/m^2 
		static double E = 3400;
		// poisson's ratio
		static double v = 0.49;
		//kg/m^3
		static double density = 1000.0;

		double penalty_weight = 100.0;
		int max_iter = 100;

		Scene scene(data_folder, input_mesh_path, target_mesh_path, fixed_points_input_mesh_path, fixed_points_target_mesh_path, implicit_input_mesh_path, implicit_target_mesh_path, correspondences_path, 
			false, num_sample_verts_subspace, true, gravity, E, v, density, penalty_weight);

		// Setup viewer
		Viewer viewer;
		viewer.setScene(&scene);
		viewer.launch();

	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}

	
}