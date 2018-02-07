/**
 * Copyright (c) 2016, David Stutz
 * Contact: david.stutz@rwth-aachen.de, davidstutz.de
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <fstream>
#include <map>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "region.h"
#include <tuple>
#include <cmath>
#include <assert.h>
#include "graph_segmentation.h"
#include <array>

/** \brief Read all image files (.png and .jpg) in the given directory.
 * \param[in] directory directory to read
 * \param[out] files found files
 */
void readDirectory(boost::filesystem::path directory,
        std::multimap<std::string, boost::filesystem::path> &files) {
    
    assert(boost::filesystem::is_directory(directory));
    
    files.clear();
    boost::filesystem::directory_iterator end;
    
    for (boost::filesystem::directory_iterator it(directory); it != end; ++it) {
        std::string extension = it->path().extension().string();
        if (extension == ".png" || extension == ".jpg" 
                || extension == ".PNG" || extension == ".JPG") {
            files.insert(std::multimap<std::string, boost::filesystem::path>::value_type(it->path().string(), it->path()));
        }
    }
}

/** \brief Write the given matrix as CSV file.
 * \param[in] file path to file to write
 * \param[in] mat matrix to write, expected to be integer matrix
 */
void writeMatCSV(boost::filesystem::path file, const cv::Mat& mat) {
    
    assert(!mat.empty());
    assert(mat.channels() == 1);
    
    std::ofstream file_stream(file.c_str());
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            file_stream << mat.at<int>(i, j);
            
            if (j < mat.cols - 1) {
                file_stream << ",";
            }
        }
        
        if (i < mat.rows  - 1) {
            file_stream << "\n";
        }
    }
    
    file_stream.close();
}

void fillRegions(const cv::Mat &image, const cv::Mat &labels, std::map<const int, Region::Region> & regions){

    for (int i = 0; i < image.rows; ++i) {
       for (int j = 0; j < image.cols; ++j) {

            const int label = labels.at<const int>(i, j);

            std::map<const int, Region::Region>::iterator it;
            it = regions.find(label);

            cv::Vec3b color_vec = image.at<cv::Vec3b>(i, j);

            int r = (int)color_vec(0);
            int g = (int)color_vec(1);
            int b = (int)color_vec(2);

            std::array<int, 3> color = {r, g, b};
             
            cv::Point pos(i, j);

            if(it != regions.end()){

                assert(it->second.label == label);

                it->second.pixels.push_back(pos);
                it->second.increment();

                std::map<std::array<int, 3>, int>::iterator it_c = it->second.colors.find(color);

                if(it_c != it->second.colors.end()){
                    it->second.colors[it_c->first] += 1;
                }
                else{
                    it->second.colors[color] = 1;
                }


            }
            else{

                Region reg;
                reg.label = label;
                reg.pixels.push_back(pos);
                reg.colors[color] = 1;

                reg.increment();
                regions[label] = reg;
            }
       }
    }
}

float regionDist(Region::Region &reg_i, Region::Region &reg_j){

    std::map<std::array<int, 3>, int> colors_i = reg_i.colors;
    std::map<std::array<int, 3>, int> colors_j = reg_j.colors;

    float dist = 0;

    for (std::map<std::array<int, 3>, int>::iterator it_i=colors_i.begin(); it_i!=colors_i.end(); ++it_i){
        
        for (std::map<std::array<int, 3>, int>::iterator it_j=colors_j.begin(); it_j!=colors_j.end(); ++it_j){
            std::array<int, 3> c_i = it_i->first;
            std::array<int, 3> c_j = it_j->first;

            float color_dist = std::sqrt(std::pow((c_i[0]-c_j[0]), 2) + std::pow((c_i[1]-c_j[1]), 2) + std::pow((c_i[2]-c_j[2]), 2));

            float freq_i = reg_i.findFrequency(reg_i, c_i);
            float freq_j = reg_j.findFrequency(reg_j, c_j);

            if (freq_j==-1 || freq_i==-1){
                std::cout<<"cannot find color in regions."<<'\n'<<std::endl;
                return -1;
            }

            dist += freq_i*freq_j*color_dist;
        }
    }
    return dist;
}

void calCenter(cv::Point &center, Region::Region &reg){

    cv::Rect br = boundingRect(cv::Mat(reg.pixels));

    float cx = (float)br.x+(float)br.width/2.0; 
    float cy = (float)br.y+(float)br.height/2.0;

    float min_dist = std::numeric_limits<int>::max();

    for (std::vector<cv::Point>::iterator it=reg.pixels.begin(); it!=reg.pixels.end(); ++it){
        int i = (*it).x;
        int j = (*it).y;

        float dist = std::sqrt(std::pow(i-cx, 2) + std::pow((j-cy), 2));
        if(dist < min_dist){
            center.x = i;
            center.y = j;
        }
    }
}

float calSaliency(std::map<const int, Region::Region> &regions, float &r_dist){

    float max = -1;

    for (std::map<const int, Region::Region>::iterator it1=regions.begin(); it1!=regions.end(); ++it1){

        std::map<std::array<int, 3>, int> colors_i = it1->second.colors;
        for (std::map<std::array<int, 3>, int>::iterator it_i=colors_i.begin(); it_i!=colors_i.end(); ++it_i){
             std::array<int, 3> c_i = it_i->first;
        }

        for (std::map<const int, Region::Region>::iterator it2=regions.begin(); it2!=regions.end(); ++it2){
            if (it1->first != it2->first){
                if (it1->second.getSaliency() == -1){

                    float dist = regionDist(it1->second, it2->second);
                    int num_pixels = it1->second.num_pixels;

                    cv::Point center1, center2;
                    calCenter(center1, it1->second);
                    calCenter(center2, it2->second);

                    float centerDist = std::sqrt(std::pow(center1.x-center2.x, 2) + std::pow(center1.y-center2.y, 2));

                    float saliency = (float)num_pixels*dist*(std::exp(-1.0*centerDist/(float)r_dist));

                    it1->second.setSaliency(saliency);
                    if (saliency > max){
                        max = saliency;
                    }
                }

            }
        }
    }
    return max;
}

/** \brief Draw the segments as contours in the image.
 * \param[in] image image to draw contours in (color image expected)
 * \param[in] labels segments to draw as integer image
 * \param[out] contours image with segments indicated by contours
 */
void drawContours(const cv::Mat & image, std::map<const int, Region::Region> & regions, cv::Mat & contours, float &max) {

    contours.create(image.rows, image.cols, CV_8UC1);

    std::map<std::array<int, 3>, int> colors;

    for (std::map<const int, Region::Region>::iterator reg=regions.begin(); reg!=regions.end(); ++reg) {
            uchar color = (float)(reg->second.getSaliency()/max)*255;

            for (std::vector<cv::Point>::iterator it=reg->second.pixels.begin(); it!=reg->second.pixels.end(); ++it){
                int i = (*it).x;
                int j = (*it).y;

                contours.at<uchar>(i, j) = color;
            }
    }
}

/** \brief Example of running graph based image segmentation for oversegmentation on
 * a directory possibly containing multiple images. Segmentations are written as CSV
 * and visualizations to the provided output directory.
 *
 * Usage:
 * 
 * \author David Stutz
 */
int main (int argc, char ** argv) {
    
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("input", boost::program_options::value<std::string>(), "folder containing the images to process")
        ("threshold", boost::program_options::value<float>()->default_value(1000.0f), "constant for threshold function")
        ("region-distance", boost::program_options::value<float>()->default_value(500.0f), "controls weight of regional distance")
        ("minimum-size", boost::program_options::value<int>()->default_value(200), "minimum component size")
        ("output", boost::program_options::value<std::string>()->default_value("output"), "save segmentation as CSV file and contour images");
    
    boost::program_options::positional_options_description positionals;
    positionals.add("input", 1);
    positionals.add("output", 1);
    
    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
    boost::program_options::notify(parameters);

    if (parameters.find("help") != parameters.end()) {
        std::cout << desc << std::endl;
        return 1;
    }
    
    boost::filesystem::path output_dir(parameters["output"].as<std::string>());
    if (!output_dir.empty()) {
        if (!boost::filesystem::is_directory(output_dir)) {
            boost::filesystem::create_directories(output_dir);
        }
    }
    
    boost::filesystem::path input_dir(parameters["input"].as<std::string>());
    if (!boost::filesystem::is_directory(input_dir)) {
        std::cout << "Image directory not found ..." << std::endl;
        return 1;
    }
    
    float threshold = parameters["threshold"].as<float>();
    float r_dist = parameters["region-distance"].as<float>();
    int minimum_segment_size = parameters["minimum-size"].as<int>();
    
    std::multimap<std::string, boost::filesystem::path> images;
    readDirectory(input_dir, images);
    
    for (std::multimap<std::string, boost::filesystem::path>::iterator it = images.begin(); 
            it != images.end(); ++it) {
        
        cv::Mat image = cv::imread(it->first);
        
        GraphSegmentationMagicThreshold magic(threshold);
        GraphSegmentationEuclideanRGB distance;
        
        GraphSegmentation segmenter;
        segmenter.setMagic(&magic);
        segmenter.setDistance(&distance);
        
        segmenter.buildGraph(image);
        segmenter.oversegmentGraph();
        segmenter.enforceMinimumSegmentSize(minimum_segment_size);
        
        cv::Mat labels = segmenter.deriveLabels();
        
        boost::filesystem::path csv_file(output_dir 
                / boost::filesystem::path(it->second.stem().string() + ".csv"));
        writeMatCSV(csv_file, labels);
        
        boost::filesystem::path contours_file(output_dir 
                    / boost::filesystem::path(it->second.stem().string() + ".png"));

        std::map<const int, Region::Region> regions;

        fillRegions(image, labels, regions);

        float max = calSaliency(regions, r_dist);

        cv::Mat image_contours;
        drawContours(image, regions, image_contours, max);

        cv::imwrite(contours_file.string(), image_contours);
    }
    
    return 0;
}
