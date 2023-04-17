#ifndef VIEWER_H
#define VIEWER_H

#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/Viewer.h>
#include <vector>

#include "scene.h"

/*
Class to abstract the libigl Viewer
in case of future replacing of the viewer
*/
class Viewer {
private:
	Scene* scene;
	void drawViewerMenu();
public:
	igl::opengl::glfw::Viewer viewer;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin;
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	unsigned int left_view, middle_view, right_view;
	int num_meshes;
	
	// meshids
	int source_mesh_id, solution_mesh_id, target_mesh_id, tumour_mesh_id;

	Viewer();
	void setScene(Scene* scene);
	//https://github.com/libigl/libigl/blob/main/include/igl/opengl/glfw/Viewer.h
	int appendMesh(unsigned int viewport_id, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
	void setMesh(unsigned int viewport_id, int mesh_id, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
	void deleteMesh(int mesh_id);
	// with fixed points
	int appendMesh(unsigned int viewport_id, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::vector<bool> fixedVerts);
	void setMesh(unsigned int viewport_id, int mesh_id, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::vector<bool> fixedVerts);

	// Set the axes in viewport adding to a mesh id
	void setAxes(unsigned int viewport_id, int mesh_id);
	// Set the points overlaid on the mesh with fixed points in a separate colour
	void setOverlayPoints(int mesh_id, const Eigen::MatrixXd& points, std::vector<bool> fixedVerts);
	// Align cameras from all the views with the given "mesh" TODO: fix this function
	void alignAllViewerCameras(Eigen::MatrixXd& V, Eigen::MatrixXi& F);
	void launch();
};

#endif
