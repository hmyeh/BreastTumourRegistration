#ifndef MATLAB_H
#define MATLAB_H

#include <igl/matlab/matlabinterface.h>
#include <Eigen/Dense>
#include <iostream>

// Store matlab stuff here
class Matlab {
private:
	// Matlab instance
	Engine* engine;

public:
	Matlab() {
		// Launch MATLAB
		igl::matlab::mlinit(&engine);
	}


	// From low to high eigenvalues
	void computeEigenDecomposition(const Eigen::SparseMatrix<double>& S, int num_ev, Eigen::MatrixXd& eigenvectors, Eigen::MatrixXd& eigenvalues) {

		// Send Sparse matrix to matlab
		igl::matlab::mlsetmatrix(&engine, "S", S);

		// Extract the first 10 eigenvectors
		igl::matlab::mleval(&engine, "[eigenvectors, eigenvalues] = eigs(S," + std::to_string(num_ev) + ", 'smallestabs')");

		// Retrieve the result
		igl::matlab::mlgetmatrix(&engine, "eigenvectors", eigenvectors);

		igl::matlab::mlgetmatrix(&engine, "eigenvalues", eigenvalues);

	}

	void computeEigenDecomposition(const Eigen::MatrixXd& S, int num_ev, Eigen::MatrixXd& eigenvectors, Eigen::MatrixXd& eigenvalues) {
		// Send Sparse matrix to matlab
		igl::matlab::mlsetmatrix(&engine, "S", S);

		// Extract the first 10 eigenvectors
		igl::matlab::mleval(&engine, "[eigenvectors, eigenvalues] = eigs(S," + std::to_string(num_ev) + ", 'smallestabs')");

		// Retrieve the result
		igl::matlab::mlgetmatrix(&engine, "eigenvectors", eigenvectors);

		igl::matlab::mlgetmatrix(&engine, "eigenvalues", eigenvalues);

	}

};

#endif
