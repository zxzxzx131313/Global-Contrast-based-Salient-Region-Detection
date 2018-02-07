#include <opencv2/opencv.hpp>
#include <map>
#include <list>
#include <array>
#include <vector>

class Region{

public: 
    Region(): label(0), num_pixels(0), saliency(-1) {
        
    };

    int label;

    int num_pixels;

    float saliency;

    std::map<std::array<int, 3>, int> colors;

    std::vector<cv::Point> pixels;

    void increment();


    float findFrequency(Region::Region &region, std::array<int, 3> &color);

    void setSaliency(float &saliency){

        this->saliency = saliency;
    }

    float getSaliency(){

        return this->saliency;
    }

    cv::Vec3b* findColor(std::map<std::array<int, 3>, int> & colors, std::array<int, 3> & color);
};
