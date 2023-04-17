#ifndef UTIL_H
#define UTIL_H

#include <Eigen/Dense>
#include <string>
#include <numeric>
#include <vector>

// TODO: maybe put this in a separate namespace

// Skew symmetric matrix cross product matrix 
// see dynamid deformables page 40 pdf
Eigen::Matrix3d getCrossProductMatrix(const Eigen::Vector3d& v);

// See dynamic deformables page 181 for rotation-variant SVD for 3D
void SVD_RV(const Eigen::Matrix3d& F, Eigen::Matrix3d& U, Eigen::Vector3d& Sigma, Eigen::Matrix3d& V);


// Get Rotation matrix from axis (u) and angle (theta) https://en.wikipedia.org/wiki/Rotation_matrix
Eigen::Matrix3d getRotationMatrix(double theta, Eigen::Vector3d u);


// Find the center of the mesh (currently by using simple average of vertices)
Eigen::Vector3d findCenterOfMesh(const Eigen::MatrixXd& verts);

// Compute point to plane distance
double computePointToPlaneDistance(Eigen::Vector3d planePoint, Eigen::Vector3d normal, Eigen::Vector3d point);

void convertToLameParameters(double E, double v, double& lambda, double& mu);

// Read file containing the fixed points as indices
std::vector<bool> readFixedVerticesFile(int num_vertices, std::string file_location);
//std::vector<bool> readFixedVerticesFile(int num_vertices, std::string);

double computeTriangleMeshVolume(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

// Method to ensure that all the normals face the correct direction
void correctTriangleMeshOrientation(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXi& oriented_F);

// Sort the given Eigen::VectorXd (without mutating) and return a list of the sorted indices
std::vector<size_t> sort_indexes(const Eigen::VectorXd& v);

// TODO: change the function to either use a custom lambda or be able to change the comparation sign
// Sort the given std::vector (without mutating) and return a list of the sorted indices
template <typename T, typename Comparator>
std::vector<size_t> sort_indexes(const std::vector<T>& v, Comparator comp) {
	//https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
	// initialize original index locations
	std::vector<size_t> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	// using std::stable_sort instead of std::sort
	// to avoid unnecessary index re-orderings
	// when v contains elements of equal values 
	std::stable_sort(idx.begin(), idx.end(),
		[&v, &comp](size_t i1, size_t i2) {return comp(v[i1], v[i2]); });

	return idx;
}


Eigen::MatrixXd nonRigidICP(double alpha, double gamma, const Eigen::MatrixXd& source_verts, const Eigen::MatrixXi& source_tri, const Eigen::MatrixXd& target_verts, const Eigen::MatrixXi& target_tri);

#endif
