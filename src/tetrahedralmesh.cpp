#include "tetrahedralmesh.h"

#include <set>

#include <igl/readOBJ.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/writeOBJ.h>
#include <igl/in_element.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/signed_distance.h>
#include <igl/barycentric_coordinates.h>
#include <igl/slice.h>
#include <igl/iterative_closest_point.h>
#include <igl/slice.h>
#include <igl/AABB.h>
#include <igl/per_face_normals.h>
#include <igl/writePLY.h>
#include <igl/signed_distance.h>

#include "util.h"


//// TriangleMesh implementations
TriangleMesh::TriangleMesh(Eigen::MatrixXd vertices, Eigen::MatrixXi triangles) : vertices(vertices), triangles(triangles) {

}

TriangleMesh TriangleMesh::readFile(std::string folder, std::string file_name) {
	Eigen::MatrixXd v;
	Eigen::MatrixXi f;
	// Load mesh
	igl::readOBJ(folder + file_name + ".obj", v, f);

	// Orienting faces for libigl renderer
	Eigen::MatrixXi oriented_faces;
	correctTriangleMeshOrientation(v, f, oriented_faces);
	return TriangleMesh(v, oriented_faces);
}

void TriangleMesh::writeFile(std::string folder, std::string file_name) {
	igl::writeOBJ(folder + file_name + ".obj", vertices, triangles);
}

double TriangleMesh::getVolume() const {
	return computeTriangleMeshVolume(vertices, triangles);
}

Eigen::Vector3d TriangleMesh::getCenter() const {
	return findCenterOfMesh(vertices);
}

void TriangleMesh::applyRigidMotion(const Eigen::Matrix3d& rotation, const Eigen::RowVector3d& translation) {
	// Apply rotation and translation to the vertices
	this->vertices = ((this->vertices * rotation).rowwise() + translation).eval();
}

void TriangleMesh::applyRotation(const Eigen::Matrix3d& rotation) {
	this->applyRigidMotion(rotation, Eigen::Vector3d::Zero());
}

void TriangleMesh::applyTranslation(const Eigen::Vector3d& translation) {
	this->applyRigidMotion(Eigen::Matrix3d::Identity(), translation);
}

const Eigen::MatrixXd& TriangleMesh::getVertices() const { return this->vertices; }
const Eigen::MatrixXi& TriangleMesh::getTriangles() const { return this->triangles; }


//// TetrahedralMesh implementations

TetrahedralMesh TetrahedralMesh::readTriangleMesh(std::string folder, std::string file_name) {
	Eigen::MatrixXd v;
	Eigen::MatrixXi f;
	// Load mesh
	igl::readOBJ(folder + file_name + ".obj", v, f);
	return TetrahedralMesh::tetrahedralize(v, f);
}

TetrahedralMesh TetrahedralMesh::tetrahedralize(const TriangleMesh& triangle_mesh) {
	return TetrahedralMesh::tetrahedralize(triangle_mesh.getVertices(), triangle_mesh.getTriangles());
}

TetrahedralMesh TetrahedralMesh::tetrahedralize(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& triangles) {
	Eigen::MatrixXd verts;
	Eigen::MatrixXi tets;
	Eigen::MatrixXi tris;
	igl::copyleft::tetgen::tetrahedralize(vertices, triangles, "pa0.001q1.414Y", verts, tets, tris);

	// Orient faces for libigl renderer
	Eigen::MatrixXi oriented_tris;
	correctTriangleMeshOrientation(verts, tris, oriented_tris);
	// Create TetrahedralMesh
	return TetrahedralMesh(verts, tets, oriented_tris);
}


TetrahedralMesh::TetrahedralMesh(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& tetrahedrons, const Eigen::MatrixXi& triangles) : vertices(vertices), tetrahedrons(tetrahedrons), triangles(triangles) {
	// https://stackoverflow.com/questions/17694579/use-stdfill-to-populate-vector-with-increasing-numbers
	std::vector<int> area(tetrahedrons.rows());
	std::iota(std::begin(area), std::end(area), 0);
	this->areas.push_back(area);
}

TetrahedralMesh::TetrahedralMesh(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& tetrahedrons, const Eigen::MatrixXi& triangles, const std::vector<std::vector<int> >& areas) :
	vertices(vertices), tetrahedrons(tetrahedrons), triangles(triangles), areas(areas) {

}

void TetrahedralMesh::addArea(const Eigen::MatrixXd& add_vertices, const Eigen::MatrixXi& add_tetrahedrons, const Eigen::MatrixXi& add_triangles) {
	// Merge the vertices of the new area with the existing vertices
	Eigen::MatrixXd merged_vertices(this->vertices.rows() + add_vertices.rows(), this->vertices.cols());
	merged_vertices << this->vertices, add_vertices;
	this->vertices = merged_vertices;

	// Merge the tetrahedrons of the new area with the existing tetrahedrons
	Eigen::MatrixXi merged_tetrahedrons(this->tetrahedrons.rows() + add_tetrahedrons.rows(), this->tetrahedrons.cols());
	merged_tetrahedrons << this->tetrahedrons, add_tetrahedrons;
	this->tetrahedrons = merged_tetrahedrons;

	// Merge the triangles of the new area with the existing triangles
	Eigen::MatrixXi merged_triangles(this->triangles.rows() + add_triangles.rows(), this->triangles.cols());
	merged_triangles << this->triangles, add_triangles;
	this->triangles = merged_triangles;

	// Create new area vector which starts counting from after the existing this->tetrahedrons
	std::vector<int> area(add_tetrahedrons.rows());
	std::iota(this->tetrahedrons.rows() + std::begin(area), this->tetrahedrons.rows() + std::end(area), this->tetrahedrons.rows());
	this->areas.push_back(area);
}

void TetrahedralMesh::writeToFile(std::string folder, std::string file_name) {
	this->writeToFile(this->vertices, folder, file_name);
}

void TetrahedralMesh::writeToFile(const Eigen::MatrixXd& x, std::string folder, std::string file_name) {
	// Sanity Check if the given matrix x has the same number of verts as the tet mesh
	assert(this->vertices.rows() == x.rows());

	std::string location = folder + file_name + ".obj";
	std::cout << "Writing Tetrahedral Mesh to: " << location << std::endl;

	// Writing only the surface vertices/triangle vertices
	Eigen::MatrixXd output_verts = this->vertices;
	std::vector<bool> ignore_verts(static_cast<unsigned long>(this->getNumVertices()), false);
	igl::slice(x, this->getSurfaceVertIndices(ignore_verts), 1, output_verts);

	Eigen::MatrixXi output_triangles = this->triangles;
	igl::writeOBJ(location, output_verts, output_triangles);
}

void TetrahedralMesh::applyRigidMotion(const Eigen::Matrix3d& rotation, const Eigen::RowVector3d& translation) {
	// Apply rotation and translation to the vertices
	this->vertices = ((this->vertices * rotation).rowwise() + translation).eval();
}

void TetrahedralMesh::applyRotation(const Eigen::Matrix3d& rotation) {
	this->applyRigidMotion(rotation, Eigen::Vector3d::Zero());
}

void TetrahedralMesh::applyTranslation(const Eigen::Vector3d& translation) {
	this->applyRigidMotion(Eigen::Matrix3d::Identity(), translation);
}

void TetrahedralMesh::rigidAlignment(TetrahedralMesh* t1, const std::vector<bool>& fixed_v1, TetrahedralMesh* t2, const std::vector<bool>& fixed_v2, ImplicitGeometry* impl_geom1, TriangleMesh* impl_geom2) {
	// Find the center of each mesh
	Eigen::Vector3d c1 = findCenterOfMesh(t1->getVertices());
	Eigen::Vector3d c2 = findCenterOfMesh(t2->getVertices());

	t1->applyTranslation(-c1);
	t2->applyTranslation(-c2);

	impl_geom1->applyTranslation(-c1);
	impl_geom2->applyTranslation(-c2);

	// v1
	Eigen::MatrixXd v1 = t1->getVertices();
	// Selector matrix for the fixed vertices
	Eigen::SparseMatrix<double> fv_selector_matrix1(v1.rows(), v1.rows());
	std::vector<Eigen::Triplet<double> > fv_selector_triplets1;
	fv_selector_triplets1.reserve(v1.rows());
	for (int i = 0; i < fixed_v1.size(); ++i) {
		if (fixed_v1[i]) {
			fv_selector_triplets1.push_back(Eigen::Triplet<double>(i, i, 1.0));
		}
	}
	fv_selector_matrix1.setFromTriplets(fv_selector_triplets1.begin(), fv_selector_triplets1.end());

	Eigen::MatrixXd fixed_verts1 = fv_selector_matrix1 * v1;

	/// t1
	Eigen::MatrixXi tri1 = t1->getTriangles();
	std::vector<Eigen::VectorXi> fixed_tri1;
	fixed_tri1.reserve(tri1.rows());
	for (int i = 0; i < tri1.rows(); ++i) {
		int count_fixed_verts = 0;
		for (int j = 0; j < tri1.cols(); ++j) {
			if (fixed_v1[tri1(i, j)]) {
				++count_fixed_verts;
			}
		}

		if (count_fixed_verts == 3) {
			fixed_tri1.push_back(tri1.row(i));
		}
	}
	Eigen::MatrixXi fixed_tri1_mat(fixed_tri1.size(), 3);
	for (int i = 0; i < fixed_tri1.size(); ++i) {
		fixed_tri1_mat.row(i) = fixed_tri1[i];
	}

	// v2
	Eigen::MatrixXd v2 = t2->getVertices();
	// Selector matrix for the fixed vertices
	Eigen::SparseMatrix<double> fv_selector_matrix2(v2.rows(), v2.rows());
	std::vector<Eigen::Triplet<double> > fv_selector_triplets2;
	fv_selector_triplets2.reserve(v2.rows());
	for (int i = 0; i < fixed_v2.size(); ++i) {
		if (fixed_v2[i]) {
			fv_selector_triplets2.push_back(Eigen::Triplet<double>(i, i, 1.0));
		}
	}
	fv_selector_matrix2.setFromTriplets(fv_selector_triplets2.begin(), fv_selector_triplets2.end());

	Eigen::MatrixXd fixed_verts2 = fv_selector_matrix2 * v2;

	/// t2
	Eigen::MatrixXi tri2 = t2->getTriangles();
	std::vector<Eigen::VectorXi> fixed_tri2;
	fixed_tri2.reserve(tri2.rows());
	for (int i = 0; i < tri2.rows(); ++i) {
		int count_fixed_verts = 0;
		for (int j = 0; j < tri2.cols(); ++j) {
			if (fixed_v2[tri2(i, j)]) {
				++count_fixed_verts;
			}
		}

		if (count_fixed_verts == 3) {
			fixed_tri2.push_back(tri2.row(i));
		}
	}
	Eigen::MatrixXi fixed_tri2_mat(fixed_tri2.size(), 3);
	for (int i = 0; i < fixed_tri2.size(); ++i) {
		fixed_tri2_mat.row(i) = fixed_tri2[i];
	}

	Eigen::Matrix3d finalRot = Eigen::Matrix3d::Identity();
	Eigen::Vector3d finaltrans;
	finaltrans.setZero();
	// Iterative closest points
	int max_iter = 100;
	for (int i = 0; i < max_iter; ++i) {
		// FOR UPDATE V1
		v2 = t2->getVertices();
		fixed_verts2 = fv_selector_matrix2 * v2;

		igl::AABB<Eigen::MatrixXd, 3> Ytree;
		Ytree.init(fixed_verts1, fixed_tri1_mat);
		Eigen::MatrixXd n1;
		igl::per_face_normals(fixed_verts1, fixed_tri1_mat, n1);

		Eigen::Matrix3d R;
		Eigen::RowVector3d t;
		igl::iterative_closest_point(fixed_verts2, fixed_tri2_mat, fixed_verts1, fixed_tri1_mat, Ytree, n1, 1000, 10, R, t);

		// Apply rotation and translation
		t2->applyRigidMotion(R, t);
		impl_geom2->applyRigidMotion(R, t);

		finalRot *= R;
		finaltrans += t;
	}

	std::cout << "FINAL ROTATION: " << std::endl;
	std::cout << finalRot << std::endl;
	std::cout << "Final Translation" << std::endl;
	std::cout << finaltrans << std::endl;
}

Eigen::VectorXi TetrahedralMesh::getSurfaceVertIndices(std::vector<bool> ignore_verts) {
	std::set<int> surface_verts_ind(this->triangles.data(), this->triangles.data() + this->triangles.size());
	// ignore verts
	for (int i = 0; i < ignore_verts.size(); ++i) {
		if (ignore_verts[i]) {
			surface_verts_ind.erase(i);
		}
	}

	Eigen::VectorXi surface_verts_vec(surface_verts_ind.size());
	for (std::set<int>::iterator it = surface_verts_ind.begin(); it != surface_verts_ind.end(); ++it) {
		surface_verts_vec(std::distance(surface_verts_ind.begin(), it)) = *it;
	}
	return surface_verts_vec;
}

Eigen::MatrixXd TetrahedralMesh::getSurfaceVertices() {
	std::vector<bool> ignore_verts = std::vector<bool>(static_cast<unsigned long>(this->vertices.rows()), false);
	Eigen::VectorXi v_ind = this->getSurfaceVertIndices(ignore_verts);
	Eigen::MatrixXd surface_verts;
	igl::slice(this->vertices, v_ind, 1, surface_verts);
	return surface_verts;
}

Eigen::MatrixXi TetrahedralMesh::getTriangles(std::vector<bool> ignore_verts) {
	Eigen::MatrixXi tri = this->triangles;
	std::vector<int> tri_idx;
	for (unsigned int i = 0; i < tri.rows(); ++i) {
		int tri_contains_vert = 0;
		for (unsigned int j = 0; j < tri.cols(); ++j) {
			if (ignore_verts[tri(i, j)]) {
				tri_contains_vert++;
			}
		}

		if (tri_contains_vert == 0) {
			tri_idx.push_back(i);
		}
	}
	Eigen::VectorXi tri_idx_vec = Eigen::Map<Eigen::VectorXi>(tri_idx.data(), tri_idx.size());

	Eigen::MatrixXi result;
	igl::slice(tri, tri_idx_vec, 1, result);
	return result;
}

double TetrahedralMesh::getVolume() const {
	return computeTriangleMeshVolume(this->vertices, this->triangles);
}

const Eigen::MatrixXd& TetrahedralMesh::getVertices() const { return this->vertices; }
const Eigen::MatrixXi& TetrahedralMesh::getTetrahedrons() const { return this->tetrahedrons; }
const Eigen::MatrixXi& TetrahedralMesh::getTriangles() const { return this->triangles; }
const std::vector<std::vector<int> >& TetrahedralMesh::getAreas() { return this->areas; }

int TetrahedralMesh::getNumVertices() const { return this->vertices.rows(); }

void TetrahedralMesh::setVertices(Eigen::MatrixXd& vertices) { this->vertices = vertices; }


//// ImplicitGeometry Implementations

ImplicitGeometry::ImplicitGeometry(const TetrahedralMesh* mesh, const TriangleMesh& implicit_mesh) : mesh(mesh), implicit_mesh(implicit_mesh) {
	this->computeMapping();
}

void ImplicitGeometry::applyRigidMotion(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation) {
	this->implicit_mesh.applyRigidMotion(rotation, translation);
}

// Applying rotation for overloading
void ImplicitGeometry::applyRotation(const Eigen::Matrix3d& rotation) {
	this->applyRigidMotion(rotation, Eigen::Vector3d::Zero());
}

// Applying translation for overloading
void ImplicitGeometry::applyTranslation(const Eigen::Vector3d& translation) {
	this->applyRigidMotion(Eigen::Matrix3d::Identity(), translation);
}

void ImplicitGeometry::computeMapping() {
	// Get the vertices and tets of the tetrahedral mesh and set to correct matrix format
	Eigen::MatrixXd V = this->mesh->getVertices();
	Eigen::MatrixXi T = this->mesh->getTetrahedrons();
	Eigen::MatrixXi TRI = this->mesh->getTriangles();

	Eigen::MatrixXd Q = this->implicit_mesh.getVertices();

	// Initialize AABB tree for the tetrahedral mesh
	igl::AABB<Eigen::MatrixXd, 3> tree;
	tree.init(V, T);

	// Initialize AABB tree for the surface mesh
	igl::AABB<Eigen::MatrixXd, 3> surface_tree;
	surface_tree.init(V, TRI);

	// Find the tets in which the vertices of the embedded geometry lie
	Eigen::VectorXi I;
	igl::in_element(V, T, Q, tree, I);

	// If point is not inside an element (-1)
	for (int i = 0; i < I.size(); ++i) {
		if (I(i) == -1) {
			// Compute closest points of implicit mesh on surface of tetrahedral mesh
			Eigen::Matrix<double, 1, 3> outside_point = Q.row(i);
			int index;
			Eigen::Matrix<double, 1, 3> c;
			surface_tree.squared_distance(V, TRI, outside_point, index, c);

			Eigen::VectorXi updated_I;
			igl::in_element(V, T, c, tree, updated_I);
			I(i) = updated_I(0);
		}
	}

	// Get the barycentric coordinates
	// https://github.com/libigl/libigl/issues/821
	Eigen::MatrixXd Va, Vb, Vc, Vd;

	// Extract to matrix tet_dims all tets indices which are named
	Eigen::VectorXi tet_dims(4);
	tet_dims << 0, 1, 2, 3;
	igl::slice(T, I, tet_dims, this->mapping_elements);

	// Extract to matrices Va, Vb, Vc, Vd each of the corner vertex positions of the corresponding tets
	Eigen::VectorXi xyz(3);
	xyz << 0, 1, 2;
	igl::slice(V, this->mapping_elements.col(0), xyz, Va);
	igl::slice(V, this->mapping_elements.col(1), xyz, Vb);
	igl::slice(V, this->mapping_elements.col(2), xyz, Vc);
	igl::slice(V, this->mapping_elements.col(3), xyz, Vd);

	igl::barycentric_coordinates(Q, Va, Vb, Vc, Vd, this->mapping_coords);
}


// Overload for the experiments to use other vertices...
TriangleMesh ImplicitGeometry::reconstructMesh(const Eigen::MatrixXd& V) const {
	if (this->mesh->getNumVertices() != V.rows()) {
		//throw exception
		throw std::invalid_argument("V does not contain the same number of vertices as the mesh used to map the implicit geometry");
	}

	// Extract to matrices Va, Vb, Vc, Vd each of the corner vertex positions
	Eigen::MatrixXd Va, Vb, Vc, Vd;
	Eigen::VectorXi xyz(3);
	xyz << 0, 1, 2;
	igl::slice(V, this->mapping_elements.col(0), xyz, Va);
	igl::slice(V, this->mapping_elements.col(1), xyz, Vb);
	igl::slice(V, this->mapping_elements.col(2), xyz, Vc);
	igl::slice(V, this->mapping_elements.col(3), xyz, Vd);

	// Interpolate the implicit vertices using barycentric coordinates
	Eigen::MatrixXd implicit_verts = Eigen::MatrixXd(this->implicit_mesh.getVertices().rows(), 3);
	for (int i = 0; i < this->mapping_coords.rows(); ++i) {
		implicit_verts.row(i) = this->mapping_coords(i, 0) * Va.row(i) + this->mapping_coords(i, 1) * Vb.row(i) + this->mapping_coords(i, 2) * Vc.row(i) +
			this->mapping_coords(i, 3) * Vd.row(i);
	}

	Eigen::MatrixXi implicit_triangles = this->implicit_mesh.getTriangles();
	return TriangleMesh(implicit_verts, implicit_triangles);
}

TriangleMesh ImplicitGeometry::reconstructMesh() const {
	return this->reconstructMesh(this->mesh->getVertices());
}

void ImplicitGeometry::writeFile(std::string folder, std::string file_name) const {
	TriangleMesh reconstructed_mesh = this->reconstructMesh();
	reconstructed_mesh.writeFile(folder, file_name);
}
