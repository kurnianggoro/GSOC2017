# [Facemark API for OpenCV][pull_request]

## Introduction

Facial landmark detection is a useful algorithm with many possible applications including expression transfer, virtual make-up, facial puppetry, faces swap, and many mores. This project aims to implement a scalable API for facial landmark detector. Furthermore, it will also implement 2 kinds of algorithms, including active appearance model and regressed local binary features.

[![Facial landmarks detection using FacemarkLBF][preview]][vid_lbf]

[See in YouTube][vid_lbf]

## Project Details
Student: Laksono kurnianggoro\
Mentors: Delia Passalacqua and Antonella Cascitelli

The works in this project is summarized in this [commits list][commits].
In the proposal, there were few list of functionalities to be added to the API.
However, during the coding period, in order to make the proposed API more reliable, various new functionalities were added as suggested by the mentors and another student who works in similar project.


#### Here are the list of functionalities that initially proposed:
- base class of the API (Facemark class) (*done*).\
This base class provides several functionalities including `read()`, `write()`, `setFaceDetector()`, `getFaces()`, `training()`, `loadModel()`, and `fit()`.
- User defined face detector (*done*)\
The configurable face detector is stored in the instance of a landmark detection algorithm and can be set using the `setFaceDetector()` function. This allows the users to use their own face detector in the algorithm.
- dataset parser (*done*).\
There are 3 kinds of utility functions that parse the information from dataset: `loadTrainingData()`, `loadDatasetList()`, and `loadFacePoints()`.
- documentation (*done*) [see preview][documentation]
- Tutorials (*done*) [see preview][tutorials]
- sample codes (*done* : 3 programs are available)
- 2 algorithms, AAM and LBF (*done*)


#### Some new functionalities that were not proposed but developed during the coding period:
- Extra parameter for the fitting function.\
Each face landmark detector might needs their own parameters in the fitting process. Some parameter are fixed in all the time but some other might changes according to the input data. In the case of fixed parameter, the developer can just add this parameter as the member of `Params`. However, in the case of dynamic parameter, the fitting function should allow extra parameter in the runtime. Hence the optional extra parameter is added to the fitting function by passing it as void parameter (`void* extra_params`) which can holds any types of parameter.
- Allows extra parameters in the user defined face detector\
Because of the same reason as the previously explained extra parameter for the fitting function, the user defined face detector also should allow extra parameter.
- Test codes\
Test codes are useful to perform automatic assessment to make sure that the implementation is compatible to various devices.
- Functionality to add the dataset one by one in the training process\n
As the discussion with another student who works on similar project, this functionality should be added since the code developer suggested that the module should not have dependency on `imgcodecs`. Previously, this dependency is needed because the dataset loading function was programmed to loads image inside the API. After this discussion, the image loading was removed from the API and the `addTrainingSample()` function is added to the base class to alleviate this problem.

#### some unsubmitted functionalities
During the coding period, there are some improvements (not listed in original plan) that were tested but failed to be merged to the OpenCV.
1. Trainer for the LBF algorithm.\
The LBF algorithm utilize liblinear to trains it regressor. During the coding period, there was an initiative to liberate the implementaion of LBF from this dependency as liblinear provides to much functionalities that are not needed. Two kinds of methods were tested to train the regressor including stochastic gradient descent (SGD) and regularized least square (RLS) regression method. However, during the test, SGD cannot produces optimal solution in general since the parameters should be set properly and depends on the characteristic of the dataset. Meanwhile for the RLS, it requires a lot of time and memory due to the needs of inverse matrix computation. Thus this method is not scalabe and will be useless for large-sized dataset.

```cpp
Mat FacemarkLBFImpl::Regressor::regressionSGD(Mat x, Mat y,
    int max_epoch, int batch_sz, double lambda, double eta){
    Mat pred;
    x.convertTo(x, CV_64F);
    y.convertTo(y, CV_64F);

    Mat w = Mat::zeros(x.cols, 1, CV_64F);
    cv::theRNG().state = cv::getTickCount();
    randn(w,0.0,1.0);
    Mat dw = Mat::zeros(x.cols, 1, CV_64F);

    std::vector<double> E;
    Mat dE;

    double gamma = 0.7;

    int maxIdx;
    for(int i=0;i<max_epoch;i++){
        pred = x*w;
        double err = norm(pred-y)/x.rows;
        if(i%10==0 || i==max_epoch-1)
            printf("%i/%i: %f\n", i, max_epoch, err);

        dE = Mat::zeros(w.rows, w.cols, CV_64F);
        for(int j=0;j<x.rows;j+=batch_sz){
            if(j+batch_sz>=x.rows){
                maxIdx = x.rows;
            }else{
                maxIdx = j+batch_sz;
            }

            Mat x_batch = Mat(x,Range(j,maxIdx));
            Mat y_batch = Mat(y,Range(j,maxIdx));

            dE = dE + x_batch.t()*(x_batch*(w-gamma*dw)-y_batch)+lambda*w;
        }//batch
        dw = gamma*dw + eta*dE/norm(dE);
        w = w - dw;
    }

    pred = x*w;
    std::cout<<sum(abs(pred-y))<<std::endl;
    std::cout<<Mat(pred-y,Range(0,30));
    return Mat(w.t()).clone();
}
```

2. Custom extra parameters for the default face detector.\
This functionality is failed to be merged and throw compilation error due problems with python wrapper.

``` cpp
struct CV_EXPORTS_W CParams{
    String cascade; //!<  the face detector
    double scaleFactor; //!< Parameter specifying how much the image size is reduced at each image scale.
    int minNeighbors; //!< Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    Size minSize; //!< Minimum possible object size.
    Size maxSize; //!< Maximum possible object size.

    CParams(
        String cascade_model,
        double sf = 1.1,
        int minN = 3,
        Size minSz = Size(30, 30),
        Size maxSz = Size()
    );
};

CV_EXPORTS_W bool getFaces( InputArray image,
                            OutputArray faces,
                            void * extra_params
                        );
```

#### Some possible improvements

#### Submission summary

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

#### some possible future improvements and un-executed ideas
1. There many possibilities to use parallelization in both implementation of LBF and AAM algorithms. Currently the parallelization is not implemented for both of them.
2. The AAM algorithm produce big-sized training model since it basically store all the face region from the dataset. An alternative to reduce this trained model size is by limiting the size of the base shape. The faces region are wrapped into the base shape. If the base shape is resized into smaller size, then each wrapped face region will be stored in smaller size.  
3. As explained in the previous part, the dependency to libnear can be eliminated.
4. The performance and processing time analysis is not yet provided. It will be useful to provide this information so that the end user could pick their setting based on the assessment data. As an example, in the LBF algorithm, the performance and processing speed changes according to number of the refinement stages, amount of depth and number of trees. Currently, the result as demonstrated in the [video][vid_lbf] works at 50~90fps for both face detection and landmark fitting. However, the user might try other setting to get their desired result. If the comparison data is provided, it will be beneficial for the user.

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
[preview]: https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/preview_lbf.gif
[facemark_api]: https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/facemark_api.png
[uml]: http://uml.mvnsearch.org/gist/334f84a4f5c59bc50aa52d1946dc1fd9
[uml_aam]: https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/facemark_aam.png
[uml_lbf]: https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/facemark_lbf.png
[pull_request]: https://github.com/opencv/opencv_contrib/pull/1257
[commits]: https://github.com/opencv/opencv_contrib/pull/1257/commits
[documentation]: http://pullrequest.opencv.org/buildbot/export/pr_contrib/1257/docs/db/dd8/classcv_1_1face_1_1Facemark.html
[tutorials]: http://pullrequest.opencv.org/buildbot/export/pr_contrib/1257/docs/d5/d47/tutorial_table_of_content_facemark.html
