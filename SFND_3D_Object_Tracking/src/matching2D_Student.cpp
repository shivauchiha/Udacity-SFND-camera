
#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    auto t = static_cast<double>(cv::getTickCount());
    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = (descriptorType == "DES_HOG") ? cv::NORM_L2 : cv::NORM_HAMMING;                        
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if(descSource.type() != CV_32F)
        { descSource.convertTo(descSource, CV_32F);
        }
      
        if(descRef.type() != CV_32F)
        { descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef,  knn_matches,2);
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
    }
  t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
    std::cout<<matcherType<<"+"<<selectorType<<" "<<matches.size()<<" matches in " << 1000 * t / 1.0 << " ms" << endl;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
     else if (descriptorType.compare("ORB") == 0)
    {
       extractor = cv::ORB::create();
    }
    
    else if (descriptorType.compare("BRIEF") == 0)
    {
       extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
       extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
       extractor = cv::AKAZE::create();
    }
    else  if(descriptorType.compare("SIFT") == 0)
    {
       

    extractor = cv::xfeatures2d::SIFT::create();
    }
  else
  { std::cout<<"descriptor unavailable"<<std::endl;
  }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}



void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
  
  if (detectorType.compare("HARRIS") == 0)
  {
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros( img.size(), CV_32FC1 );
    double t = (double)cv::getTickCount();
    cv::cornerHarris( img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT );
    cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    cv::convertScaleAbs( dst_norm, dst_norm_scaled );
    int minResponse = 100;
    
    for(size_t i =0;i<dst_norm.rows;i++)
    { for(size_t j=0 ; j<dst_norm.cols;j++)
    {
      int tempintensity = (int)dst_norm.at<float>(i,j);
      if(tempintensity>minResponse)
      { cv::KeyPoint tempkeys;
        tempkeys.pt = cv::Point2f(j,i);
        tempkeys.size = 2*apertureSize;
        tempkeys.response = tempintensity;
       
        bool overlap_flag = false;
        for(auto it = keypoints.begin();it != keypoints.end();++it)
        {if(cv::KeyPoint::overlap(*it,tempkeys)>0.0)
        { overlap_flag = true;
          if((*it).response<tempkeys.response)
          {*it=tempkeys;
           break;
          }
        }
        }
        if(!overlap_flag)
        {keypoints.push_back(tempkeys);
        }
        
      }
      
    }
    }
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
  }
  
  
  if (detectorType.compare("FAST") == 0)
  { double t = (double)cv::getTickCount();
    cv::FAST(img,keypoints,30,true,cv::FastFeatureDetector::TYPE_9_16);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  }
  
  if (detectorType.compare("BRISK") == 0)
    
  { cv::Ptr<cv::BRISK> detector =  cv::BRISK::create();
    double t = (double)cv::getTickCount();
    detector->detect(img,keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  }
  
  if (detectorType.compare("ORB") == 0)
    
  { 
    cv::Ptr<cv::ORB> detector =  cv::ORB::create();
    double t = (double)cv::getTickCount();
    detector->detect(img,keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "ORB detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  }
  
  if (detectorType.compare("AKAZE") == 0)
    
  { 
    cv::Ptr<cv::AKAZE> detector =  cv::AKAZE::create();
    double t = (double)cv::getTickCount();
    detector->detect(img,keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "AKAZE detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  }
  
  if (detectorType.compare("SIFT") == 0)
    
  { 
    cv::Ptr<cv::xfeatures2d::SIFT> detector =  cv::xfeatures2d::SIFT::create();
    double t = (double)cv::getTickCount();
    detector->detect(img,keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "SIFT detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  }

  
  
  if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType+" Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }



}
