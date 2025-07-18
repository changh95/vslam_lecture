#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main() {
    cout << "Eigen Tutorial for Visual SLAM" << endl;
    cout << "=============================" << endl << endl;

    // Basic matrix operations
    cout << "1. Basic Matrix Operations:" << endl;
    Matrix3d mat33 = Matrix3d::Random();  // 3x3 double matrix with random values
    Vector3d vec3(1, 2, 3);               // 3D vector

    cout << "3x3 Matrix:\n" << mat33 << "\n\n";
    cout << "3D Vector:\n" << vec3 << "\n\n";
    cout << "Matrix * Vector:\n" << mat33 * vec3 << "\n\n";

    // Essential operations for SLAM
    cout << "2. Essential SLAM Operations:" << endl;
    
    // Matrix creation #1 - Identity
    Matrix3d rotation1 = Matrix3d::Identity();
    cout << "Identity rotation matrix:\n" << rotation1 << "\n\n";
    
    // Matrix creation #2 - Explicit creation
    Matrix3d rotation2;
    rotation2 << 1, 0, 0,
                0, 1, 0,
                0, 0, 1;
    cout << "Manually set rotation matrix:\n" << rotation2 << "\n\n";
    
    // Translation vector
    Vector3d translation(1, 2, 3);
    cout << "Translation Vector:\n" << translation << "\n\n";

    // Transformation matrix (SE3)
    Matrix4d T = Matrix4d::Identity();
    T.block<3,3>(0,0) = rotation1;
    T.block<3,1>(0,3) = translation;
    cout << "Transformation Matrix (SE3):\n" << T << "\n\n";

    // Point transformation
    Vector3d point(1, 0, 0);
    Vector3d transformed_point = rotation1 * point + translation;
    cout << "Original Point:\n" << point << "\n\n";
    cout << "Transformed Point:\n" << transformed_point << "\n\n";

    // Matrix operations commonly used in optimization
    cout << "3. Matrix Operations for Optimization:" << endl;
    Matrix3d A = Matrix3d::Random();
    cout << "Matrix A:\n" << A << "\n\n";
    cout << "Determinant: " << A.determinant() << "\n";
    cout << "Inverse:\n" << A.inverse() << "\n\n";
    cout << "Transpose:\n" << A.transpose() << "\n\n";

    return 0;
}
