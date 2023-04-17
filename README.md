# BreastTumourRegistration


## Building the project

Dependencies
- Eigen https://gitlab.com/libeigen/eigen
- Libigl https://github.com/libigl/libigl
- Glfw https://github.com/glfw/glfw
- Glad https://github.com/Dav1dde/glad
- Tetgen https://github.com/libigl/tetgen
- Autodiff https://github.com/autodiff/autodiff
- Intel MKL libraries (https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html)
- Matlab (https://nl.mathworks.com/help/matlab/matlab_external/build-c-engine-programs.html)
- imgui (https://github.com/ocornut/imgui)
- libigl-imgui (https://github.com/libigl/libigl-imgui)
- boost (https://www.boost.org/)
- gmp (https://gmplib.org/)
- mpfr (https://www.mpfr.org/)
- cgal (https://www.cgal.org/)


The dependencies are either installed with vcpkg (https://github.com/microsoft/vcpkg) or self-built/header-only libraries and placed in the folder `external` in the root of this directory.
Any compiled libraries are used x64-windows

Libraries installed with vcpkg :
- boost
- cgal
- gmp
- mpfr

Libraries placed under folder `external`:
- autodiff
- eigen
- glad
- glfw
- imgui
- libigl (libigl-imgui is placed inside this library)
- tetgen

Compiled libraries:
- glfw
- glad
- tetgen
- igl with cgal and opengl(glad/glfw)
- intel mkl
- matlab
