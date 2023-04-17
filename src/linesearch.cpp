#include "linesearch.h"
#include <iostream>


//// LineSearchFunction implementation

LineSearchFunction::LineSearchFunction(EnergyFunction* energy_function, Eigen::Ref<Eigen::Matrix3Xd> x, Eigen::Ref<Eigen::VectorXd> descentDir) : energy_function(energy_function), x(x), descentDir(descentDir) {
    // Change the view of the descent direction matrix to be compatible with this->x
    descentDirMat = Eigen::Map<Eigen::MatrixXd>(descentDir.data(), x.rows(), x.cols());
}

double LineSearchFunction::evaluateFunction(double alpha) {
    Eigen::MatrixXd predictX = x + alpha * descentDirMat;
    double value = this->energy_function->computeEnergy(predictX);
    if (std::isnan(value)) {
        value = std::numeric_limits<double>::infinity();
    }
    return value;
}

double LineSearchFunction::evaluateFunctionGradient(double alpha) {
    Eigen::MatrixXd predictX = x + alpha * descentDirMat;
    return this->energy_function->computeGradient(predictX).transpose() * descentDir;
}

double LineSearchFunction::directionInfinityNorm() {
    return descentDir.lpNorm<Eigen::Infinity>();
}

//// SNHLinesearch implementation

// This code is taken from Stable Neohookean flesh simulation paper
void SNHLinesearch::MinimizeInSearchDirection(EnergyFunction* energy_function, Eigen::Ref<Eigen::Matrix3Xd> x, Eigen::Ref<Eigen::VectorXd> direction, double& alphaMin, double& Umin) {
    LineSearchFunction evalFunction(energy_function, x, direction);
    // Width of the bracket
    constexpr double tol = 1.0e-4;

    // Middle point in bracket
    double alpha_middle = 0.0;
    double U_middle = evalFunction.evaluateFunction(alpha_middle);

    // Right bracket on minimum
    double alpha_right = 1.0;
    //mesh.UpdatePerturbedCachedState(direction);
    double U_right = evalFunction.evaluateFunction(alpha_right);

    // Left bracket on minimum
    double alpha_left;
    double U_left;

    if (U_middle > U_right)
    {
        U_left = U_middle;
        alpha_left = alpha_middle;
        U_middle = U_right;
        alpha_middle = alpha_right;

        // Search for an alpha with energy greater than the middle
        while (U_right <= U_middle)
        {
            alpha_right *= 2.0;
            U_right = evalFunction.evaluateFunction(alpha_right);
        }
    }
    else if (U_middle < U_right)
    {
        alpha_left = -1.0;
        U_left = evalFunction.evaluateFunction(alpha_left);

        // Search for an alpha with energy greater than the middle
        while (U_left <= U_middle)
        {
            alpha_left *= 2.0;
            U_left = evalFunction.evaluateFunction(alpha_left);
        }
    }
    else // (U_middle == U_right)
    {
        assert(U_middle == U_right);
        // Try the other direction
        alpha_left = -1.0;
        U_left = evalFunction.evaluateFunction(alpha_left);
        if (U_left != U_middle)
        {
            if (U_left > U_middle)
            {
                std::cerr << "Right gives same energy...";
                std::cerr << "and left step has greater energy... this corner case isn't coded up yet." << std::endl;
                std::cerr << "U_left, U_center, U_right: " << U_left << "    " << U_middle << "    " << U_right << std::endl;
                std::exit(EXIT_FAILURE);
            }
            else // U_left < U_middle
            {
                // Shift the values to the right
                U_right = U_middle;
                alpha_right = alpha_middle;
                U_middle = U_left;
                alpha_middle = alpha_left;

                // Search for an alpha with energy greater than the middle
                alpha_left = -2.0;
                U_left = evalFunction.evaluateFunction(alpha_left);
                while (U_left <= U_middle)
                {
                    alpha_left *= 2.0;
                    U_left = evalFunction.evaluateFunction(alpha_left);
                }
            }
        }
        // U_left == U_middle == U_right
        // else
        // {
        //     std::cout << "and left step gives same energy. Probably a flat energy landscape. Searching anyway." << std::endl;
        // }
    }

    while ((alpha_right - alpha_left) > tol)
    {
        assert(alpha_left <= alpha_middle);
        assert(alpha_right >= alpha_middle);
        assert(U_left >= U_middle);
        assert(U_right >= U_middle);

        // Recurse in the right subinterval
        if ((alpha_right - alpha_middle) > (alpha_middle - alpha_left))
        {
            const double alpha_new = 0.5 * (alpha_middle + alpha_right);
            assert(alpha_new > alpha_middle); assert(alpha_new < alpha_right);
            const double U_new = evalFunction.evaluateFunction(alpha_new);
            // If this is the new energy minimum, tighten the left bound
            if (U_new < U_middle)
            {
                alpha_left = alpha_middle;
                U_left = U_middle;
                alpha_middle = alpha_new;
                U_middle = U_new;
                assert(U_left >= U_middle);
                assert(U_right >= U_middle);
            }
            // Otherwise, tighten the right bound
            else
            {
                alpha_right = alpha_new;
                U_right = U_new;
                assert(U_left >= U_middle);
                assert(U_right >= U_middle);
            }
        }
        // Recurse in the left subinterval
        else
        {
            const double alpha_new = 0.5 * (alpha_left + alpha_middle);
            assert(alpha_new > alpha_left); assert(alpha_new < alpha_middle);

            const double U_new = evalFunction.evaluateFunction(alpha_new);
            // If this is a new energy minimum, tighten the right bound
            if (U_new < U_middle)
            {
                alpha_right = alpha_middle;
                U_right = U_middle;
                alpha_middle = alpha_new;
                U_middle = U_new;
                assert(U_left >= U_middle);
                assert(U_right >= U_middle);
            }
            // Otherwise, tighten the left bound
            else
            {
                alpha_left = alpha_new;
                U_left = U_new;
                assert(U_left >= U_middle);
                assert(U_right >= U_middle);
            }
        }
    }

    alphaMin = alpha_middle;
    Umin = U_middle;
}

