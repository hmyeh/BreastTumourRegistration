#ifndef TETRAHEDRAL_MESH_H
#define TETRAHEDRAL_MESH_H

#include <Eigen/Dense>
#include <vector>
#include <numeric>
#include <string>
#include <stdexcept>


class ImplicitGeometry;

class TriangleMesh {
private:
	Eigen::MatrixXd vertices;
	Eigen::MatrixXi triangles;
public:
	TriangleMesh() {}
	TriangleMesh(Eigen::MatrixXd vertices, Eigen::MatrixXi triangles);

	// Obj file only
	static TriangleMesh readFile(std::string folder, std::string file_name);

	void writeFile(std::string folder, std::string file_name);

	double getVolume() const;
	Eigen::Vector3d getCenter() const;

	void applyRigidMotion(const Eigen::Matrix3d& rotation, const Eigen::RowVector3d& translation);
	// Applying rotation for overloading
	void applyRotation(const Eigen::Matrix3d& rotation);

	// Applying translation for overloading
	void applyTranslation(const Eigen::Vector3d& translation);

	const Eigen::MatrixXd& getVertices() const;
	const Eigen::MatrixXi& getTriangles() const;
};


// Tetrahedral mesh
class TetrahedralMesh {
private:
	Eigen::MatrixXd vertices;
	Eigen::MatrixXi tetrahedrons;
	Eigen::MatrixXi triangles;

	// Storing all the areas
	std::vector<std::vector<int> > areas;
	// Area in this context is part of the mesh/tetrahedron elements which should be modeled with different parameters (such as other material models/parameters)

public:
	static TetrahedralMesh readTriangleMesh(std::string folder, std::string file_name);

	static TetrahedralMesh tetrahedralize(const TriangleMesh& triangle_mesh);
	static TetrahedralMesh tetrahedralize(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& triangles);


	// No vertices/tets and no areas constructor (this might be needed for pointers, but also for if we want to give separate meshes so afterwards call addArea function)
	TetrahedralMesh() {}

	// Single defined area constructor
	TetrahedralMesh(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& tetrahedrons, const Eigen::MatrixXi& triangles);
	// Multiple defined areas constructor
	TetrahedralMesh(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& tetrahedrons, const Eigen::MatrixXi& triangles, const std::vector<std::vector<int> >& areas);

	// Adds a new area to the existing tetrahedral mesh
	void addArea(const Eigen::MatrixXd& add_vertices, const Eigen::MatrixXi& add_tetrahedrons, const Eigen::MatrixXi& add_triangles);

	void writeToFile(std::string folder, std::string file_name);
	// Currently only version to write the triangle mesh and not the tetrahedral mesh itself
	// Used for debugging of solver.cpp
	void writeToFile(const Eigen::MatrixXd& x, std::string folder, std::string file_name);

	// Apply rigid motion to the tetrahedral mesh
	void applyRigidMotion(const Eigen::Matrix3d& rotation, const Eigen::RowVector3d& translation);
	// Applying rotation for overloading
	void applyRotation(const Eigen::Matrix3d& rotation);
	// Applying translation for overloading
	void applyTranslation(const Eigen::Vector3d& translation);

	// Rigid alignment between two tetrahedral meshes (for now making it static) also alignment between the fixed points only
	// Note: it is assumed that the only constraints given are the fixed point constraints TODO: maybe check if the constraint is a fixed point constraint class
	static void rigidAlignment(TetrahedralMesh* t1, const std::vector<bool>& fixed_v1, TetrahedralMesh* t2, const std::vector<bool>& fixed_v2, ImplicitGeometry* impl_geom1, TriangleMesh* impl_geom2);

	// overloaded function to ignore fixed vertices
	Eigen::VectorXi getSurfaceVertIndices(std::vector<bool> ignore_verts);
	Eigen::MatrixXd getSurfaceVertices();
	Eigen::MatrixXi getTriangles(std::vector<bool> ignore_verts);

	// Getters
	double getVolume() const;
	const Eigen::MatrixXd& getVertices() const;
	const Eigen::MatrixXi& getTetrahedrons() const;
	const Eigen::MatrixXi& getTriangles() const;
	const std::vector<std::vector<int> >& getAreas();
	int getNumVertices() const;

	void setVertices(Eigen::MatrixXd& vertices);
};



class ImplicitGeometry {
private:
	const TetrahedralMesh* mesh;

	TriangleMesh implicit_mesh;

	Eigen::MatrixXd mapping_coords;
	// tetrahedrons in this case
	Eigen::MatrixXi mapping_elements;
public:
	ImplicitGeometry() {}

	ImplicitGeometry(const TetrahedralMesh* mesh, const TriangleMesh& implicit_mesh);

	void applyRigidMotion(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation);
	// Applying rotation for overloading
	void applyRotation(const Eigen::Matrix3d& rotation);
	// Applying translation for overloading
	void applyTranslation(const Eigen::Vector3d& translation);

	void computeMapping();

	// Overload for the experiments to use other vertices
	TriangleMesh reconstructMesh(const Eigen::MatrixXd& V) const;
	TriangleMesh reconstructMesh() const;

	void writeFile(std::string folder, std::string file_name) const;

};


#endif
