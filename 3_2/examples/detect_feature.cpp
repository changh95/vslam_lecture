#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " left_image_path right_image_path" << std::endl;
        return 1;
    }

    const std::string left_image_path = argv[1];
    const std::string right_image_path = argv[2];

    if (!std::filesystem::exists(left_image_path) || !std::filesystem::exists(right_image_path)) {
        std::cerr << "Image files do not exist!" << std::endl;
        return 1;
    }

    // Read images
    cv::Mat img_left = cv::imread(left_image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img_right = cv::imread(right_image_path, cv::IMREAD_GRAYSCALE);

    if (img_left.empty() || img_right.empty()) {
        std::cerr << "Failed to read images" << std::endl;
        return 1;
    }

    // Initialize OpenCV objects
    const auto feature_detector = cv::ORB::create(1000);
    const auto bf_matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    const cv::FlannBasedMatcher knn_matcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    
    // Detect features
    std::vector<cv::KeyPoint> kpts_left, kpts_right;
    cv::Mat desc_left, desc_right;

    feature_detector->detectAndCompute(img_left, cv::Mat(), kpts_left, desc_left);
    feature_detector->detectAndCompute(img_right, cv::Mat(), kpts_right, desc_right);

    if (desc_left.empty() || desc_right.empty()) {
        std::cerr << "No features detected in images" << std::endl;
        return 1;
    }

    // Brute-force matching
    std::vector<cv::DMatch> bf_matches;
    bf_matcher->match(desc_left, desc_right, bf_matches);
    
    constexpr int distance_thresh = 50;
    std::vector<cv::DMatch> good_bf_matches;
    good_bf_matches.reserve(bf_matches.size());
    for (const auto &match : bf_matches) {
        if (match.distance < distance_thresh) {
            good_bf_matches.push_back(match);
        }
    }

    // KNN matching
    std::vector<std::vector<cv::DMatch>> knn_matches;
    knn_matcher.knnMatch(desc_left, desc_right, knn_matches, 2);

    constexpr float ratio_thresh = 0.8f;
    std::vector<cv::DMatch> good_knn_matches;
    good_knn_matches.reserve(knn_matches.size());
    for (const auto &match : knn_matches) {
        if (match.size() == 2 && match[0].distance < ratio_thresh * match[1].distance) {
            good_knn_matches.push_back(match[0]);
        }
    }

    // Draw matches
    cv::Mat img_bf, img_knn;
    cv::drawMatches(img_left, kpts_left, img_right, kpts_right, good_bf_matches,
                   img_bf, cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::drawMatches(img_left, kpts_left, img_right, kpts_right, good_knn_matches,
                   img_knn, cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("BF Matches", img_bf);
    cv::imshow("KNN Matches", img_knn);
    cv::waitKey(0);

    return 0;
}