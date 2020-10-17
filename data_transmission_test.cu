#include "../include/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    CHECK(cudaSetDevice(dev));
    string filename = "./data/pic.jpeg";
    double maxTime = 0;
    double minTime = 1000;
    double averageTime = 0;
    double sum = 0;
    int num = 1;
//    bool flag = 1;

    for(int i = 0; i < num;i++){


        Mat img = imread(filename,cv::IMREAD_GRAYSCALE);
//    imshow("original img",img);
//    waitKey(0);

        int height = img.rows;
        int width = img.cols;
        int size = height * width;
//    int nbytes = 122738;//pic的实际大小
//    int nbytes = 262159;//Lena
        int nbytes = size * sizeof(uchar);

//仅显示一次信息
//        if(flag) {
//            cudaDeviceProp deviceProp;
//            CHECK(cudaGetDeviceProperties(&deviceProp, dev));
//
//            printf("%s starting at ", argv[0]);
//            printf("device %d: %s memory size %d nbyte %5.2fMB\n height %d,width %d \n", dev,
//                   deviceProp.name, size, nbytes / (1024.0f * 1024.0f), height, width);
//            flag = 0;
//        }

        uchar *h_array = (uchar *)malloc(nbytes);
        uchar *d_array = 0;

        double iStart = seconds();
        CHECK(cudaMalloc((void **)&d_array, nbytes));
        double iElaps = seconds() - iStart;
        printf(" back and forth  elapsed  %f sec \n",iElaps);



//    将图像矩阵转为在cpu上的一维数组
        for (int i = 0; i < height; i++)
        {
            uchar *data = img.ptr<uchar>(i);
            for (int j = 0; j < width; j++)
            {
                h_array[i*width+j] = data[j];
            }
        }

        CHECK(cudaMemcpy(d_array, h_array, nbytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(h_array,d_array, nbytes, cudaMemcpyDeviceToHost));


//把一维数组从gpu传输到cpu后，在cpu上显示图像
        Mat img_h(height,width,CV_8UC1);
        for (int i = 0; i < height; i++)
        {
            uchar *p = img_h.ptr<uchar>(i);
            for (int j = 0; j < width; j++)
            {
                p[j] = h_array[i * width + j];
            }
        }
//    imshow("img in CPU",img_h);
//    waitKey(0);


        CHECK(cudaFree(d_array));
        free(h_array);


        sum = sum + iElaps;
        if (iElaps > maxTime)   maxTime = iElaps;
        if (iElaps < minTime)   minTime = iElaps;

       //printf(" back and forth  elapsed  %f sec \n",iElaps);

    }
    CHECK(cudaDeviceReset());
    averageTime = sum / num;
    printf(" back and forth  elapsed average time %f sec\n max time %f sec\n min time %f sec \n", averageTime,maxTime,minTime);
    return 0;
}

