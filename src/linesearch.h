#ifndef LINESEARCH_H
#define LINESEARCH_H

#include <Eigen/Dense>
#include "energy.h"
#include <limits>
#include "optimizationmethods.h"


//https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/line_search.cc
class LineSearchFunction {
private:
	EnergyFunction* energy_function;
	Eigen::Matrix3Xd x;
	Eigen::VectorXd descentDir;
	Eigen::MatrixXd descentDirMat;
public:
	LineSearchFunction(EnergyFunction* energy_function, Eigen::Ref<Eigen::Matrix3Xd> x, Eigen::Ref<Eigen::VectorXd> descentDir);

	// Energy function evaulation / obscuring the workings of energy from the rest of linesearch
	// Evaluate the line search objective
	//
	//   f(x) = p(position + alpha * direction)
	double evaluateFunction(double alpha);
	double evaluateFunctionGradient(double alpha);
	double directionInfinityNorm();
};


// Used from Stable Neohookean reference code with minimal changes
class SNHLinesearch {
public:
	static void MinimizeInSearchDirection(EnergyFunction* energy_function, Eigen::Ref<Eigen::Matrix3Xd> x, Eigen::Ref<Eigen::VectorXd> direction, double& alphaMin, double& Umin);


};

#endif
