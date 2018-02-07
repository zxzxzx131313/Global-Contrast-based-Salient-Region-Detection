#include <opencv2/opencv.hpp>
#include "region.h"
#include <map>

#include <iostream>

float Region::findFrequency(Region::Region &region, std::array<int, 3> &color){

    std::map<std::array<int, 3>, int>::iterator it=region.colors.find(color);

    if(it != region.colors.end()){
        int num_occur = it->second;
        return (float)num_occur / (float) region.num_pixels;
    }

    std::cout << "color DNE, cannot find frequency." <<std::endl;
    return -1;
}

void Region::increment(){
    this->num_pixels+=1;
}