//
//  main.cpp
//  HoughTransform
//
//  Created by Angelo Gonzalez on 12/17/14.
//  Copyright (c) 2014 Angelo Gonzalez. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;
//polar line equation: d = ccos(theta) - rsin(theta)

float euclidean(Mat &histogram1, Mat &histogram2) {
    float distance = 0;
    
    for (int x = 0; x < histogram1.rows; x++) {
        distance += (histogram1.at<float>(x) - histogram2.at<float>(x)) * (histogram1.at<float>(x) - histogram2.at<float>(x));
    }
    
    return sqrt(distance);
}

float manhattan(Mat &histogram1, Mat &histogram2) {
    float distance = 0;
    
    for (int x = 0; x < histogram1.rows; x++) {
        distance += abs(histogram1.at<float>(x) - histogram2.at<float>(x));
    }
    
    return distance;
}

//how I'm storing the point list
struct Coordinate {
    int x;
    int y;
};

//to sort vectors by size() with sort()
struct CapacityGreater : public binary_function<vector<Coordinate>, vector<Coordinate>, bool>
{
    bool operator()(const vector<Coordinate> &a, const vector<Coordinate> &b) const {
        return a.size() > b.size();
    }
};

//takes in the original unaltered color image, number of lines to draw, percentage of magnitude to threshold by (0 to 1), number of steps for d,
//and number of steps for theta
Mat houghTransform(Mat &input, int numLines, double thresholdPercent, int binStepD, int binStepTheta) {
    
    //histogram to count the horizontal, vertical, and diagonal
    Mat histogram = Mat(3, 1, CV_32F, 0.0);
    Mat greyImage, image;
    cvtColor(input, image, CV_BGR2GRAY);
    
    //Noise Reduction
    GaussianBlur(image, greyImage, Size(3, 3), 1.5);

    //compute the max possible d for each image and then calculate the number of bins for the accumulator array
    int maxSizeD = sqrt(pow(greyImage.rows, 2) + pow(greyImage.cols, 2));
    int binsD = maxSizeD / binStepD;
    int binsTheta = 360 / binStepTheta;
    
    //3D vector to store a vector of Coordinate structs
    vector<vector<vector<Coordinate>>> accArray;
    
    accArray.resize(binsD);
    for (int i = 0; i < accArray.size(); i++) {
        accArray[i].resize(binsTheta);
    }
    
    //calculate gradients
    Mat gradX = Mat(greyImage.rows, greyImage.cols, CV_32F, 0.0);
    Mat gradY = Mat(greyImage.rows, greyImage.cols, CV_32F, 0.0);
    Sobel(greyImage, gradX, CV_32F, 1, 0, 3);
    Sobel(greyImage, gradY, CV_32F, 0, 1, 3);
    
    //calculate mags
    Mat sum = Mat(greyImage.rows, greyImage.cols, CV_64F);
    Mat prodX = Mat(greyImage.rows, greyImage.cols, CV_64F);
    Mat prodY = Mat(greyImage.rows, greyImage.cols, CV_64F);
    multiply(gradX, gradX, prodX);
    multiply(gradY, gradY, prodY);
    sum = prodX + prodY;
    sqrt(sum, sum);
    
    Mat mag = sum.clone();
    
    //get the maximum magnitude and calculate the threshold based on the percentage given
    double maxMag;
    minMaxIdx(mag, 0, &maxMag);
    int threshold = maxMag * thresholdPercent;
    
    //calculate slopes
    Mat slopes = Mat(greyImage.rows, greyImage.cols, CV_32F, 0.0);
    divide(gradY, gradX, slopes);
    
    
    //calculate angles
    Mat theta = Mat(slopes.rows, slopes.cols, CV_32F, 0.0);
    
    for (int x = 0; x < slopes.rows; x++) {
        for (int y = 0; y < slopes.cols; y++) {
            
            //for easier range checking
            float currentDirection = atan(slopes.at<float>(x, y)) * (180 / 3.142);
            
            //get full range of angles 0 to 360 this is what will be used for theta
            int currentDir360 = atan2(gradY.at<float>(x, y), gradX.at<float>(x, y)) * (180 / M_PI);
            
            if(currentDirection < 0) currentDirection += 180;
            
            if(currentDir360 < 0) currentDir360 += 360;
            
            theta.at<float>(x, y) = (currentDir360);
            
            //increase the count of horizontal, vertical and diagonal lines in their respective bins if it passes the threshold
            if (mag.at<float>(x, y) > threshold) {
                if ((currentDirection > 22.5 && currentDirection <= 67.5) || (currentDirection > 112.5 && currentDirection <= 157.5)) {
                    histogram.at<float>(2) += 1;
                }
                else if (currentDirection > 67.5 && currentDirection <= 112.5) {
                    histogram.at<float>(1) += 1;
                }
                else {
                    histogram.at<float>(0) += 1;
                }
            }
        }
    }
    
    //calculate d
    //polar line equation: d = c * cos(theta) - r * sin(theta)
    Mat d = Mat(theta.rows, theta.cols, CV_32F, 0.0);
    
    for (int x = 0; x < theta.rows; x++) {
        for (int y = 0; y < theta.cols; y++) {
            
            //cosine and sine functions take in radians so convert angle to radians to compute the proper d
            float computedD = abs(y * cos(theta.at<float>(x, y) * (M_PI / 180)) - x * sin(theta.at<float>(x, y) * (M_PI / 180)));
            
            d.at<float>(x, y) = computedD;
        }
    }
    
    //quantize the data and accumulate the array
    for (int x = 0; x < theta.rows; x++) {
        for (int y = 0; y < theta.cols; y++) {
            
            //threshold for the magnitude
            if (mag.at<float>(x, y) < threshold) {
                continue;
            }
            
            //if theta is 360 make it 0
            int angle = theta.at<float>(x, y);
            if (angle == 360) {
                angle = 0;
            }
            //cout << "row: " << x << "col: " << y << "d: " << d.at<float>(x, y) << "theta: " << angle << endl;
            
            int D = d.at<float>(x, y);
            
            //accumulate and quantize
            Coordinate pt;
            pt.x = y;
            pt.y = x;
            
            accArray.at(D / binStepD).at(angle / binStepTheta).push_back(pt);
        }
    }
    
    //4 main lines we care about
    Coordinate line1Start, line1End, line2Start, line2End, line3Start, line3End, line4Start, line4End;
    
    //count the number of lines and push any Coordinate vectors that exist into the 2D vector list to be sorted
    vector<vector<Coordinate>> list;
    
    int totalNumLines = 0;
    for (int x = 0; x < accArray.size(); x++) {
        for (int y = 0; y < accArray[0].size(); y++) {
            
            if (accArray.at(x).at(y).size() != 0) {
                totalNumLines++;
                list.push_back(accArray.at(x).at(y));
            }
        }
    }
    
    //sort the 2D vector of Coodinates by their weights (or size())
    sort(list.begin(), list.end(), CapacityGreater());
    
    line1Start = list[0].front();
    line1End = list[0].back();
    line2Start = list[1].front();
    line2End = list[1].back();
    line3Start = list[2].front();
    line3End = list[2].back();
    line4Start = list[3].front();
    line4End = list[3].back();
    
    
    Point l1p1(line1Start.x, line1Start.y);
    Point l1p2(line1End.x, line1End.y);
    Point l2p1(line2Start.x, line2Start.y);
    Point l2p2(line2End.x, line2End.y);
    Point l3p1(line3Start.x, line3Start.y);
    Point l3p2(line3End.x, line3End.y);
    Point l4p1(line4Start.x, line4Start.y);
    Point l4p2(line4End.x, line4End.y);


    line(input, l1p1, l1p2, Scalar(0,0,255), 2, CV_AA);
    line(input, l2p1, l2p2, Scalar(0,255,0), 2, CV_AA);
    line(input, l3p1, l3p2, Scalar(255,0,0), 2, CV_AA);
    line(input, l4p1, l4p2, Scalar(127,127,127), 2, CV_AA);
    
    //if there are more lines that were specified to be drawn in the parameters, then draw more as white lines
    if (numLines > 4) {
        for (int i = 4; i < numLines; i++) {
            Point p1(list.at(i).front().x, list.at(i).front().y);
            Point p2(list.at(i).back().x, list.at(i).back().y);
            
        
            line(input, p1, p2, Scalar(255, 255, 255), 2, CV_AA);
            
        }
    }
    
    imshow("Hough Transform", input);
    waitKey();
    
    //divide each histogram bin by the total number of lines found
    for (int i = 0;  i < histogram.rows; i++) {
        histogram.at<float>(i) = histogram.at<float>(i) / totalNumLines;
    }
    
    return histogram;
}

int main(int argc, const char * argv[]) {
    
    string inFileName = argv[1];
    ifstream inFile(inFileName.c_str(), ios_base::in); // creates an input file stream
    
    string queryFileName;
    inFile >> queryFileName;  // reads the first line of the file into queryFileName
    
    // make histogram(s) from the image with the name: queryFileName
    Mat queryInput = imread(queryFileName);
    Mat houghHistogram;
    
    houghHistogram = houghTransform(queryInput, 15, .25, 4, 4);
    
    //maps for storing distances
    map<float, string> L1Hough;
    map<float, string> L2Hough;

    while (!inFile.eof())
    {
        string anotherFileName;
        inFile >> anotherFileName;
        
        // make histogram(s) from the image with the name: anotherFileName
        // use the histogram(s) made from the queryFileName and the ones from
        // anotherFileName and compare them with L1, L2, etc.
        Mat anotherInput = imread(anotherFileName);
        Mat histogram;
        
        histogram = houghTransform(anotherInput, 15, .25, 4, 4);
        
        L1Hough[manhattan(houghHistogram, histogram)] = anotherFileName;
        L2Hough[euclidean(houghHistogram, histogram)] = anotherFileName;
    }
    
    printf("%s \n", "L1 Hough Transform");
    printf("%s %s \n", "Query Image: ", queryFileName.c_str());
    //Intersection for color histograms
    for(map<float,string>::iterator it=L1Hough.begin(); it!=L1Hough.end(); ++it) {
        
        printf("%10f %s %50s \n", (*it).first , ": ", (*it).second.c_str());
        
    }
    
    cout << endl;
    
    printf("%s \n", "L2 Hough Transform");
    printf("%s %s \n", "Query Image: ", queryFileName.c_str());
    //Intersection for color histograms
    for(map<float,string>::iterator it=L2Hough.begin(); it!=L2Hough.end(); ++it) {
        
        printf("%10f %s %50s \n", (*it).first , ": ", (*it).second.c_str());
        
    }
    
    cout << endl;
    
    
    
    return 0;
}
