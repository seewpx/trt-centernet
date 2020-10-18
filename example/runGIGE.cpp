//
// Created by xu on 20-10-16.
//


#include <argparse.h>
#include <string>
#include <iostream>
#include <memory>
#include "ctdetNet.h"
#include "utils.h"

#define VIS_WIDTH 1264*2
#define VIS_HEITH 1500*2
#define VIS_X (4096 - VIS_WIDTH)/2
#define VIS_Y (3000 - VIS_HEITH)/2

int main(int argc, const char** argv){
    optparse::OptionParser parser;
    parser.add_option("-e", "--input-engine-file").dest("engineFile").set_default("test.engine")
            .help("the path of onnx file");
    parser.add_option("-c", "--input-video-file").dest("capFile");
    optparse::Values options = parser.parse_args(argc, argv);
    if(options["engineFile"].size() == 0){
        std::cout << "no file input" << std::endl;
        exit(-1);
    }

    cv::RNG rng(244);
    std::vector<cv::Scalar> color = { cv::Scalar(255, 0,0),cv::Scalar(0, 255,0)};
    //for(int i=0; i<ctdet::classNum;++i)color.push_back(randomColor(rng));


    cv::namedWindow("result",cv::WINDOW_NORMAL);
    cv::resizeWindow("result",1024,768);

    ctdet::ctdetNet net(options["engineFile"]);
    std::unique_ptr<float[]> outputData(new float[net.outputBufferSize]);

    cv::Mat frame ;
    cv::Mat img;
    cv::VideoCapture cap;
    if(options["capFile"].size()>0) 
        cap.open(options["capFile"]);
    else
    {
        cap.open(0, cv::CAP_ARAVIS);
        cap.set(cv::CAP_PROP_EXPOSURE, 0.01);
        cap.set(cv::CAP_PROP_GAIN, 10);
        cap.set(cv::CAP_PROP_FPS, 60);
        cap.set(cv::CAP_PROP_FOURCC, 'GRBG');
    }
        
        
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    
    while (cap.read(frame))
    {
        if (frame.empty()) {
            std::cerr << "ERROR! blank frame grabbed\n";
            continue;
        }
        if (options["capFile"].size()>0) {
            img = std::move(frame);
        }
        else {
            cv::Rect myROI(VIS_X, VIS_Y, VIS_WIDTH, VIS_HEITH);
            cv::Mat croppedRef(frame, myROI);
            cv::resize(croppedRef, img, cv::Size(0,0), 0.25, 0.25);
        }
        
        auto inputData = prepareImage(img,net.forwardFace);

        net.doInference(inputData.data(), outputData.get());
        net.printTime();

        int num_det = static_cast<int>(outputData[0]);

        std::vector<Detection> result;

        result.resize(num_det);

        memcpy(result.data(), &outputData[1], num_det * sizeof(Detection));

        postProcess(result,img,net.forwardFace);

        drawImg(result,img,color,net.forwardFace);

        cv::imshow("result",img);
        if((cv::waitKey(1)& 0xff) == 27){
            cv::destroyAllWindows();
            return 0;
        };
        //cv::waitKey(0);

    }


    return 0;
}
