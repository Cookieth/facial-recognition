#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>

// Video Capture Documentation: https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
// XML Haar Cascade for Facial Recognition: https://github.com/opencv/opencv/tree/master/data/haarcascades

// Classifier for Haar Cascade, as per objdetect.hpp
cv::CascadeClassifier face_cascade;

int main(int, char **)
{
    cv::Mat frame;

    // ===== Vieo Capture Setup =====
    cv::VideoCapture cap;

    // open the default camera using default API
    // cap.open(0);
    // OR advance usage: select any API backend
    int deviceID = 0;        // 0 = open default camera
    int apiID = cv::CAP_ANY; // 0 = autodetect default API

    // open selected camera using selected API
    cap.open(deviceID, apiID);

    // check if we succeeded
    if (!cap.isOpened())
    {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    // ===== Facial Recognition Setup =====
    // Learn more: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

    face_cascade.load("./harcascade_frontalface_default.xml");

    //--- GRAB AND WRITE LOOP
    std::cout << "Start grabbing" << std::endl
              << "Press any key to terminate" << std::endl;

    for (;;)
    {

        cap.read(frame);

        if (frame.empty())
        {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        cv::Mat frame_gray;
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(frame_gray, frame_gray);

        std::vector<cv::Rect> faces;

        // Uses the aforementioned classifier, and creates a vector for faces
        // Documentation: https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
        face_cascade.detectMultiScale(frame_gray, faces);

        for (size_t i = 0; i < faces.size(); i++)
        {
            // Doc: https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9
            cv::rectangle(frame, faces[i], cv::Scalar(66, 134, 245));
        }
        //-- Show what you got
        cv::imshow("Capture - Face detection", frame);

        // Live display without Bounding Box
        //imshow("Live", frame);

        if (cv::waitKey(5) >= 0)
            break;
    }

    return 0;
}
