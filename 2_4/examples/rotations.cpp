#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;

int main() {
    cout << "3D Rotation Representations Tutorial" << endl;
    cout << "=================================" << endl << endl;

    // 1. Rotation Matrix (SO3)
    cout << "1. Rotation Matrix (SO3):" << endl;
    Matrix3d R = Matrix3d::Identity();
    // Create rotation of 45 degrees around Z axis
    double angle = M_PI / 4;
    R = AngleAxisd(angle, Vector3d::UnitZ());
    cout << "Rotation matrix (45° around Z):\n" << R << "\n\n";

    // 2. Axis-Angle representation
    cout << "2. Axis-Angle representation:" << endl;
    AngleAxisd aa(R);  // Convert rotation matrix to axis-angle
    cout << "Angle: " << aa.angle() * 180 / M_PI << " degrees" << endl;
    cout << "Axis: " << aa.axis().transpose() << "\n\n";

    // 3. Euler Angles (RPY - Roll Pitch Yaw)
    cout << "3. Euler Angles (RPY):" << endl;
    Vector3d euler = R.eulerAngles(2, 1, 0); // ZYX order
    cout << "Roll: " << euler[2] * 180 / M_PI << " degrees" << endl;
    cout << "Pitch: " << euler[1] * 180 / M_PI << " degrees" << endl;
    cout << "Yaw: " << euler[0] * 180 / M_PI << " degrees\n\n";

    // 4. Quaternion
    cout << "4. Quaternion:" << endl;
    Quaterniond q(R);  // Convert rotation matrix to quaternion
    cout << "w: " << q.w() << endl;
    cout << "x: " << q.x() << endl;
    cout << "y: " << q.y() << endl;
    cout << "z: " << q.z() << "\n\n";

    // 5. Sophus SO3
    cout << "5. Sophus SO3:" << endl;
    Sophus::SO3d so3 = Sophus::SO3d::rotZ(angle);  // Create SO3 from angle
    cout << "SO3 matrix:\n" << so3.matrix() << "\n\n";

    // 6. SE3 (3D Rigid Body Motion)
    cout << "6. SE3 (3D Rigid Body Motion):" << endl;
    Vector3d t(1, 2, 3);  // Translation vector
    Sophus::SE3d se3(R, t);  // Create SE3 from rotation and translation
    cout << "SE3 matrix:\n" << se3.matrix() << "\n\n";

    // 7. Converting between representations
    cout << "7. Converting between representations:" << endl;
    // Quaternion -> Rotation Matrix -> Euler Angles
    Matrix3d R_from_q = q.toRotationMatrix();
    Vector3d euler_from_q = R_from_q.eulerAngles(2, 1, 0);
    cout << "Euler angles from quaternion:\n" << euler_from_q.transpose() * 180 / M_PI << "\n\n";

    // 8. Rotation composition
    cout << "8. Rotation composition:" << endl;
    // Create two rotations and compose them
    Quaterniond q1 = Quaterniond(AngleAxisd(M_PI/4, Vector3d::UnitX()));
    Quaterniond q2 = Quaterniond(AngleAxisd(M_PI/4, Vector3d::UnitY()));
    Quaterniond q_composed = q1 * q2;
    cout << "Composed rotation matrix:\n" << q_composed.toRotationMatrix() << endl;

    return 0;
}
