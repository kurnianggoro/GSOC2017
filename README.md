# [Facemark API for OpenCV][pull_request]

## Introduction

Facial landmark detection is a useful algorithm with many possible applications including expression transfer, virtual make-up, facial puppetry, faces swap, and many mores. This project aims to implement a scalable API for facial landmark detector. Furthermore, it will also implement 2 kinds of algorithms, including active appearance model and regressed local binary features.

[![Facial landmarks detection using FacemarkLBF][preview]][vid_lbf]

[See in YouTube][vid_lbf]

This [API][pull_request] extends the face module in current OpenCV contrib repository.
The list of files in this extension is listed as follows:


| Component                      | Remarks                                                                                   |
| ------------------------------ | ----------------------------------------------------------------------------------------- |
| *include/opencv2/face:*         | The required header files                                                                 |
| facemark.hpp               | Header for the base class of Facemark API                                                 |
| facemarkAAM.hpp            | Header for the AAM algorithm                                                              |
| facemarkLBF.hpp            | Header for the LBF algorithm                                                              |
| *src:*                          | Implementations of the algorithm are stored in this folder                                |
| facemark.cpp               | Contains the implementation of several utility functions                                  |
| facemarkAAM.cpp            | Contains the implementation of the AAM algorithm                                          |
| facemarkLBF.cpp            | Contains the implementation of the LBF algorithm                                          |
| *tutorials:*                    | Tutorials for the users                                                                   |
| facemark_tutorial.markdown | The main menu                                                                             |
| facemark_add_algorithm     | Tutorial on how to extends the API with a new facial landmark detection algorithm         |
| facemark_usage             | Tutorial on how to use the API                                                            |
| *samples:*                      | Several sample codes are provided as the reference for the users                          |
| facemark_demo_aam.cpp      | Demo for the AAM algorithm                                                                |
| facemark_demo_lbf.cpp      | Demo for the LBF algorithm                                                                |
| facemark_lbf_fitting.cpp   | Demo to detect facial landmarks on video                                                  |
| *test:*                         | Test code to make sure that the implementations are able to works in various environments |
| test_facemark.cpp          | Test the utility functions                                                                |
| test_facemark_aam.cpp      | Test the implementation of AAM algorithm                                                  |
| test_facemark_lbf.cpp      | Test the implementation of LBF algorithm                                                  |





## The Facemark API
![UML of the clasess in Facemark API][facemark_api]

#### Some explanations
bla bla bal

#### example of code for each functions
```c++
std::vector<cv::Rect> roi;
cv::CascadeClassifier face_cascade("haarcascade_frontalface_alt.xml");
cv::face::getFacesHaar(frame, roi, face_cascade);
for(int j=0;j<rects.size();j++){
    cv::rectangle(frame, rects[j], cv::Scalar(255,0,255));
}
cv::imshow("detection", frame);
```

- explanation for the annotation format
- information of dataset, etc

#### some possible future improvements
asd asd asd

## The Facemark AAM algorithm
![UML of the FacemarkAAM][uml_aam]


#### Some explanations
bla bla bal

#### example of usage
```c++
std::vector<cv::Rect> roi;
cv::CascadeClassifier face_cascade("haarcascade_frontalface_alt.xml");
cv::face::getFacesHaar(frame, roi, face_cascade);
for(int j=0;j<rects.size();j++){
    cv::rectangle(frame, rects[j], cv::Scalar(255,0,255));
}
cv::imshow("detection", frame);
```

#### some possible future improvements
asd asd asd

## The Facemark LBF algorithm
![UML of the FacemarkLBF][uml_lbf]

#### Some explanations
bla bla bal

#### example of usage
```c++
std::vector<cv::Rect> roi;
cv::CascadeClassifier face_cascade("haarcascade_frontalface_alt.xml");
cv::face::getFacesHaar(frame, roi, face_cascade);
for(int j=0;j<rects.size();j++){
    cv::rectangle(frame, rects[j], cv::Scalar(255,0,255));
}
cv::imshow("detection", frame);
```

#### some possible future improvements
change the regression algorithm


[vid_lbf]: https://www.youtube.com/watch?v=B7WGyhl2zm8
[preview]: data/preview_lbf.gif
[facemark_api]: data/facemark_api.png
[uml]: http://uml.mvnsearch.org/gist/334f84a4f5c59bc50aa52d1946dc1fd9
[uml_aam]: data/facemark_aam.png
[uml_lbf]: data/facemark_lbf.png
[pull_request]: https://github.com/opencv/opencv_contrib/pull/1257
