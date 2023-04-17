#include "util.h"
#include <iostream> 
#include <fstream>
#include <igl/volume.h>

#include <igl/orient_outward.h>
#include <igl/bfs_orient.h>

// For now its here the definition of PI
#define PI 3.14159265

Eigen::Matrix3d getCrossProductMatrix(const Eigen::Vector3d& v) {
	Eigen::Matrix3d vHat;
	vHat << 0, -v(2), v(1),
		v(2), 0, -v(0),
		-v(1), v(0), 0;
	return vHat;
}


void SVD_RV(const Eigen::Matrix3d& F, Eigen::Matrix3d& U, Eigen::Vector3d& Sigma, Eigen::Matrix3d& V) {
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Sigma = svd.singularValues();
	U = svd.matrixU();
	V = svd.matrixV();

	if (U.determinant() < 0.0)
	{
		U.col(0) *= -1.0;
		Sigma(0) *= -1.0;
	}
	if (V.determinant() < 0.0)
	{
		V.col(0) *= -1.0;
		Sigma(0) *= -1.0;
	}

}


Eigen::Matrix3d getRotationMatrix(double theta, Eigen::Vector3d u) {
	return cos(theta * PI / 180.0)* Eigen::Matrix3d::Identity() + sin(theta * PI / 180.0) * getCrossProductMatrix(u) +  (1 - cos(theta * PI / 180.0)) * u * u.transpose();
}


// Currently computing the simple average to find the center of the mesh
Eigen::Vector3d findCenterOfMesh(const Eigen::MatrixXd& verts) {
	return verts.colwise().mean();
}

double computePointToPlaneDistance(Eigen::Vector3d planePoint, Eigen::Vector3d normal, Eigen::Vector3d point) {
	// Normalize normal just in case
	normal.normalize();
	return (point - planePoint).dot(normal);
}



void convertToLameParameters(double E, double v, double& mu, double& lambda) {
	mu = E / (2.0 * (1.0 + v));
	lambda = (E * v) / ((1.0 + v) * (1.0 - 2.0 * v));
}


std::vector<bool> readFixedVerticesFile(int num_vertices, std::string file_location) {
	std::vector<bool> fixed_vertices(num_vertices);
	std::ifstream file(file_location);

	if (!file)
	{
		std::cerr << "Unable to open input file: " << file_location << std::endl;
		throw std::exception("Unable to open input file");
	}

	// Read the file
	//std::vector<int> fixed_verts;
	std::string number_as_string;
	while (std::getline(file, number_as_string, ' '))
	{
		int vert_idx = std::stoi(number_as_string);
		// check that the number is smaller than the number of verts
		assert(vert_idx < num_vertices);
		//fixed_verts.push_back(vert_idx);
		fixed_vertices[vert_idx] = true;

	}

	return fixed_vertices;
}

//https://github.com/libigl/libigl/issues/694
double computeTriangleMeshVolume(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
	Eigen::MatrixXd V2(V.rows() + 1, V.cols());
	V2.topRows(V.rows()) = V;
	V2.bottomRows(1).setZero();
	Eigen::MatrixXi T(F.rows(), 4);
	T.leftCols(3) = F;
	T.rightCols(1).setConstant(V.rows());
	Eigen::VectorXd vol;
	igl::volume(V2, T, vol);
	return std::abs(vol.sum());
}


void correctTriangleMeshOrientation(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXi& oriented_F) {
	Eigen::MatrixXi FF;
	Eigen::MatrixXi C;

	igl::bfs_orient(F, FF, C);

	Eigen::VectorXi I;
	igl::orient_outward(V, FF, C, oriented_F, I);
}


std::vector<size_t> sort_indexes(const Eigen::VectorXd& v) {
	//https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
	// initialize original index locations
	std::vector<size_t> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	// using std::stable_sort instead of std::sort
	// to avoid unnecessary index re-orderings
	// when v contains elements of equal values 
	std::stable_sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v(i1) < v(i2); });

	return idx;
}

#include <igl/edges.h>

// Adapted from https://github.com/Tonsty/Non-Rigid-Registar/blob/master/src/optimal_nonrigid_icp/optimal_nonrigid_icp.h
Eigen::MatrixXd nonRigidICP(double alpha, double gamma, const Eigen::MatrixXd& source_verts, const Eigen::MatrixXi& source_tri, const Eigen::MatrixXd& target_verts, const Eigen::MatrixXi& target_tri) {
	Eigen::MatrixXi edges;
	igl::edges(source_tri, edges);

	int n = source_verts.rows();
	int m = edges.rows();

	Eigen::SparseMatrix<double> A(4 * m + n, 4 * n);

	std::vector< Eigen::Triplet<double> > alpha_M_G;
	for (int i = 0; i < m; ++i)
	{
		int a = edges(i, 0);
		int b = edges(i, 1);

		for (int j = 0; j < 3; j++) alpha_M_G.push_back(Eigen::Triplet<double>(i * 4 + j, a * 4 + j, alpha));
		alpha_M_G.push_back(Eigen::Triplet<double>(i * 4 + 3, a * 4 + 3, alpha * gamma));

		for (int j = 0; j < 3; j++) alpha_M_G.push_back(Eigen::Triplet<double>(i * 4 + j, b * 4 + j, -alpha));
		alpha_M_G.push_back(Eigen::Triplet<double>(i * 4 + 3, b * 4 + 3, -alpha * gamma));
	}
	std::cout << "alpha_M_G calculated!" << std::endl;


	// WEIGHTS
	std::vector<double> weights(n);
	for (int i = 0; i < weights.size(); ++i) weights[i] = 1.0;

	std::vector< Eigen::Triplet<double> > W_D;
	for (int i = 0; i < n; ++i)
	{
		Eigen::Vector3d point = source_verts.row(i);

		double weight = weights[i];

		for (int j = 0; j < 3; ++j) W_D.push_back(Eigen::Triplet<double>(4 * m + i, i * 4 + j, weight * point(j)));
		W_D.push_back(Eigen::Triplet<double>(4 * m + i, i * 4 + 3, weight));
	}
	std::cout << "W_D calculated!" << std::endl;

	std::vector< Eigen::Triplet<double> > _A = alpha_M_G;
	_A.insert(_A.end(), W_D.begin(), W_D.end());
	std::cout << "_A calculated!" << std::endl;

	A.setFromTriplets(_A.begin(), _A.end());
	std::cout << "A calculated!" << std::endl;

	Eigen::MatrixX3d B = Eigen::MatrixX3d::Zero(4 * m + n, 3);
	for (int i = 0; i < n; ++i)
	{
		Eigen::Vector3d point = target_verts.row(i);

		double weight = weights[i];
		for (int j = 0; j < 3; j++) B(4 * m + i, j) = weight * point(j);
	}
	std::cout << "B calculated!" << std::endl;


	Eigen::SparseMatrix<double> ATA = Eigen::SparseMatrix<double>(A.transpose()) * A;
	std::cout << "ATA calculated!" << std::endl;
	Eigen::MatrixX3d ATB = Eigen::SparseMatrix<double>(A.transpose()) * B;
	std::cout << "ATB calculated!" << std::endl;

	Eigen::ConjugateGradient< Eigen::SparseMatrix<double> > solver;
	solver.compute(ATA);
	std::cout << "solver computed ATA!" << std::endl;
	if (solver.info() != Eigen::Success)
	{
		std::cerr << "Decomposition failed" << std::endl;
		//return;
	}

	Eigen::MatrixX3d X = solver.solve(ATB);
	std::cout << "X calculated!" << std::endl;

	Eigen::MatrixXd updated_verts(source_verts.rows(), source_verts.cols());
	Eigen::Matrix3Xd XT = X.transpose();
	for (int i = 0; i < n; ++i)
	{
		Eigen::Vector3d xyz = source_verts.row(i);
		Eigen::Vector4d point(xyz(0), xyz(1), xyz(2), 1.0);
		Eigen::Vector3d point_transformed = XT.block<3, 4>(0, 4 * i) * point;
		updated_verts.row(i) = point_transformed;

		Eigen::VectorXd txyz = target_verts.row(i);

	}
	return updated_verts;
}

