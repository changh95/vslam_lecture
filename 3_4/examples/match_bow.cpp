#include "DBoW2/DBoW2.h"
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " image1 image2 image3 query_image" << std::endl;
    return 1;
  }

  std::vector<cv::Mat> images(3);
  for (int i = 0; i < 3; i++) {
    images[i] = cv::imread(argv[i+1], cv::IMREAD_GRAYSCALE);
    if (images[i].empty()) {
      std::cerr << "Failed to read image: " << argv[i+1] << std::endl;
      return 1;
    }
  }

  const auto feature_detector = cv::ORB::create();
  std::vector<std::vector<cv::Mat>> v_descriptors;

  // Feature extraction
  for (auto &img : images) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    feature_detector->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
    v_descriptors.push_back(std::vector<cv::Mat>());
    v_descriptors.back().resize(descriptors.rows);

    for (int i = 0; i < descriptors.rows; i++) {
      v_descriptors.back()[i] = descriptors.row(i);
    }
  }

  // Vocabulary creation
  const int k = 9;
  const int L = 3;
  const DBoW2::WeightingType weight = DBoW2::TF_IDF;
  const DBoW2::ScoringType score = DBoW2::L1_NORM;

  OrbVocabulary voc(k, L, weight, score);
  voc.create(v_descriptors);
  voc.save("vocabulary.yml.gz");

  // Global feature database creation
  OrbDatabase db(voc, false, 0);
  for (int i = 0; i < images.size(); i++) {
    db.add(v_descriptors[i]);
  }

  // Query
  cv::Mat query_img = cv::imread(argv[4], cv::IMREAD_GRAYSCALE);
  std::vector<cv::KeyPoint> query_kpts;
  cv::Mat query_desc;
  feature_detector->detectAndCompute(query_img, cv::Mat(), query_kpts,
                                     query_desc);
  std::vector<cv::Mat> v_query_desc;
  v_query_desc.resize(query_desc.rows);
  for (int i = 0; i < query_desc.rows; i++) {
    v_query_desc[i] = query_desc.row(i);
  }

  DBoW2::QueryResults ret;
  db.query(v_query_desc, ret, 4);
  std::cout << "Searching for image "
            << ": " << ret << std::endl;

  cv::imshow("query", query_img);
  cv::imshow("result", images[ret[0].Id]);
  cv::waitKey(0);
}