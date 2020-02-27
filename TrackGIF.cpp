/*
  a simple code for video edit
*/

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/video/tracking.hpp>
#include <sys/time.h>
#include <vector>
#include <string>
#include <Magick++.h>

using namespace cv;
using namespace std;
using namespace Magick;

#define SCALE 0.5      // frame scale for tracking
#define RATIO 0.65     // frame tracking weight

#define RATIO_0 0.2    // frame tracking weight
#define RATIO_1 0.5    // frame tracking weight
#define RATIO_2 0.3    // frame tracking weight
#define GIF_HEIGHT 400   // GIF image height for resize ?
#define GIF_WIDTH 300    // GIF image width for resize ?
#define GIF_SCALE 0.7    // GIF image resize scale
//#define DEBUG 1        // print some log information to debug code
//#define SAVE_VIDEO 1   // is save edited GIF video? if don't save it just commit it

vector<string> trackerTypes = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"}; 

cv::Mat Magick2Mat(Image& image){
    // Get dimensions of Magick++ Image
    int w = image.columns();
    int h = image.rows();
    // Make OpenCV Mat of same size with 8-bit and 3 channels
    Mat opencvImage(h,w,CV_8UC3);
    // Unpack Magick++ pixels into OpenCV Mat structure
    image.write(0,0,w,h,"BGR",Magick::CharPixel,opencvImage.data);
    return opencvImage;
}

// create tracker by name
Ptr<Tracker> createTrackerByName(string trackerType) {
    Ptr<Tracker> tracker;
    if (trackerType ==  trackerTypes[0])
        tracker = TrackerBoosting::create();
    else if (trackerType == trackerTypes[1])
        tracker = TrackerMIL::create();
    else if (trackerType == trackerTypes[2])
        tracker = TrackerKCF::create();
    else if (trackerType == trackerTypes[3])
        tracker = TrackerTLD::create();
    else if (trackerType == trackerTypes[4])
        tracker = TrackerMedianFlow::create();
    else if (trackerType == trackerTypes[5])
        tracker = TrackerGOTURN::create();
    else if (trackerType == trackerTypes[6])
        tracker = TrackerMOSSE::create();
    else if (trackerType == trackerTypes[7])
        tracker = TrackerCSRT::create();
    else {
        cout << "Incorrect tracker name" << endl;
        cout << "Available trackers are: " << endl;
        for (vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it)
            std::cout << " " << *it << endl;
    }
    return tracker;
}

// Fill the vector with random colors
void getRandomColors(vector<Scalar> &colors, int numColors){
    RNG rng(0);
    for(int i=0; i < numColors; i++)
        colors.push_back(Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)));
}

int main(int argc, char * argv[]){
    struct timeval start, end;
    cout << "Default tracking algoritm is CSRT" << endl;
    cout << "Available tracking algorithms are:" << endl;
    for (vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it)
        std::cout << " " << *it << endl;
    // KalmanFilter initialation
    KalmanFilter KF(4,2);
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(.1));
    Mat measurement = Mat::zeros(2, 1, CV_32F);
    cv::Mat state(4, 1, CV_32F);
    
    // read gif picture to warp
    // Initialise ImageMagick library
    string gifPath = argv[2];
    InitializeMagick("");
    
    // Create Magick++ Image object and read image file
    list<Image> imageList;
    readImages(&imageList, gifPath);
    cv::Mat gifFrame, rz_gifFrame, gray_gifFrame;
    std::vector<cv::Mat> im_warp_vector, imGray_warp_vector;
    
    for(list<Image>::iterator it = imageList.begin(); it != imageList.end(); it++){
        gifFrame = Magick2Mat(*it);
        if(gifFrame.cols>GIF_WIDTH && gifFrame.rows>GIF_HEIGHT)
            cv::resize(gifFrame, rz_gifFrame, cv::Size(0,0), GIF_SCALE, GIF_SCALE, INTER_CUBIC);
        else
            rz_gifFrame = gifFrame.clone();

        im_warp_vector.push_back(rz_gifFrame.clone());
        cv::cvtColor(rz_gifFrame, gray_gifFrame, CV_BGR2GRAY);
        imGray_warp_vector.push_back(gray_gifFrame.clone());
    }
#ifdef DEBUG
    cout << "GIF frame numbers : " << im_warp_vector.size() << ", " << imGray_warp_vector.size() << endl;
#endif
    cv::Point half_warp = cv::Point(gray_gifFrame.cols/2, gray_gifFrame.rows/2);
    
    // set default values for tracking algorithm and video
    // Set tracker type. Change this to try different trackers.
    string trackerType = "MEDIANFLOW";
    string videoPath = argv[1];
    
    // Initialize MultiTracker with tracking algo
    vector<Rect> bboxes;
    
    // create a video capture object to read videos
    cv::VideoCapture cap(videoPath);
    Mat frame;
    
    // quit if unabke to read video file
    if(!cap.isOpened())
    {
        cout << "Error opening video file " << videoPath << endl;
        return -1;
    }
    
    // read first frame
    cap >> frame;
    cv::Mat Tframe;
    cv::transpose(frame, Tframe);
    cv::flip(Tframe, Tframe, 1);
    Mat rz_frame;
    resize(frame, rz_frame, cv::Size(0,0), SCALE, SCALE, INTER_CUBIC);
    // draw bounding boxes over objects
    // selectROI's default behaviour is to draw box starting from the center
    // when fromCenter is set to false, you can draw box starting from top left corner
    bool showCrosshair = true;
    bool fromCenter = false;
    cout << "\n==========================================================\n";
    cout << "OpenCV says press c to cancel objects selection process" << endl;
    cout << "It doesn't work. Press Escape to exit selection process" << endl;
    cout << "argv[1]: Video file path" << endl;
    cout << "argv[2]: GIF file path" << endl;
    cout << "argv[3]: Save video name" << endl;
    cout << "\n==========================================================\n";
    cv::selectROIs("GIF_Tracking", frame, bboxes, showCrosshair, fromCenter);
    
    // quit if there are no objects to track
    if(bboxes.size() < 1)
        return 0;

    std::vector<cv::Scalar> colors;
    getRandomColors(colors, bboxes.size());
    
    // Create multitracker
    Ptr<MultiTracker> multiTracker = cv::MultiTracker::create();
    
    vector<cv::Point2f> prev_frame, prev_prev_frame;
    // initialize multitracker
    for(int i=0; i < bboxes.size(); i++){
        cv::Point tl = bboxes[i].tl()*SCALE;
        cv::Point br = bboxes[i].br()*SCALE;
        Rect temp = Rect(tl, br);
        bboxes[i] = temp;
        prev_frame.push_back( (temp.br()+temp.tl())*0.5);
        prev_prev_frame.push_back( (temp.br()+temp.tl())*0.5);
        multiTracker->add(createTrackerByName(trackerType), rz_frame, Rect2d(bboxes[i]));
    }
    KF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
//    Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));
    
    KF.statePre.at<float>(0) = prev_frame[0].x;
    KF.statePre.at<float>(1) = prev_frame[0].y;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    // process video and track objects
    cout << "\n==========================================================\n";
    cout << "Started tracking, press ESC to quit." << endl;
    Rect objRect;
    cv::Point2f current_frame;
    int frameID = 1;
    int gifID = 0;
    int frame_Height = frame.rows;
    int frame_Width = frame.cols;

#ifdef SAVE_VIDEO
    double fps = cap.get(CV_CAP_PROP_FPS);
    VideoWriter writer(argv[3], CV_FOURCC('M', 'J', 'P', 'G'), fps, Size(frame_Width, frame_Height));
#endif
    
    while(cap.isOpened()){
        // get frame from the video
        cap >> frame;
//        // fix strange video
//        if (frame.empty()) break;
//        if(frameID == 1){
//            cv::transpose(frame, Tframe);
//            cv::flip(Tframe, Tframe, 1);
//            // cv::imwrite("debug.jpg", Tframe);
//        }
        
        // stop the program if reached end of video
        if (frame.empty()) break;

        resize(frame, rz_frame, cv::Size(0,0), SCALE, SCALE, INTER_CUBIC);
        //update the tracking result with new frame
        gettimeofday(&start, NULL);
        multiTracker->update(rz_frame);

        // draw tracked objects
        for(unsigned i=0; i<multiTracker->getObjects().size(); i++){
            //rectangle(frame, multiTracker->getObjects()[i], colors[i], 2, 1);
            //跳帧更新 update objRect
            objRect = multiTracker->getObjects()[i];
            
            if(frameID<2){
                current_frame.x = RATIO*prev_frame[i].x + (1-RATIO)*(objRect.br().x+objRect.tl().x)*0.5;
                current_frame.y = RATIO*prev_frame[i].y + (1-RATIO)*(objRect.br().y+objRect.tl().y)*0.5;
            }
            else{
                current_frame.x = RATIO_0*prev_prev_frame[i].x + RATIO_1*prev_frame[i].x + RATIO_2*(objRect.br().x+objRect.tl().x)*0.5;
                current_frame.y = RATIO_0*prev_prev_frame[i].y + RATIO_1*prev_frame[i].y + RATIO_2*(objRect.br().y+objRect.tl().y)*0.5;
            }
            measurement.at<float>(0) = current_frame.x;
            measurement.at<float>(1) = current_frame.y;
            Mat estimated = KF.correct(measurement);
            cv::Point kfPred;
            state = KF.predict();
            if(state.at<float>(0) !=0 || state.at<float>(1) !=0){
                kfPred.x = min(max(0, (int)state.at<float>(0)), rz_frame.cols);
                kfPred.y = min(max(0, (int)state.at<float>(1)), rz_frame.rows);
                // circle(frame, debug/SCALE, 9, Scalar(0,0,255));
            }
            current_frame.x = 0.15*kfPred.x + (1-0.15)*current_frame.x;
            current_frame.y = 0.15*kfPred.y + (1-0.15)*current_frame.y;
            
            Rect deScaleRect;
            cv::Point deScaleRectStart;
            deScaleRectStart.x = (current_frame.x - objRect.width/2)/SCALE;
            deScaleRectStart.y = (current_frame.y - objRect.height/2)/SCALE;
            deScaleRect = Rect(deScaleRectStart.x, deScaleRectStart.y, objRect.width/2/SCALE, objRect.height/2/SCALE);

            cv::Point center_of_objRect;
            center_of_objRect.x = (int)(current_frame.x/SCALE);
            center_of_objRect.y = (int)(current_frame.y/SCALE);
            if(center_of_objRect.x < half_warp.x+1){
                center_of_objRect.x = half_warp.x+1;
                if(center_of_objRect.y < half_warp.y+1){
                    center_of_objRect.y = half_warp.y+1;
                }
                if(center_of_objRect.y > (frame_Height - half_warp.y-1) ){
                    center_of_objRect.y = frame_Height - half_warp.y-1;
                }
            }
            else{
                if(center_of_objRect.x > (frame_Width - half_warp.x-1)){
                    center_of_objRect.x = frame_Width - half_warp.x-1;
                }
                if(center_of_objRect.y < half_warp.y+1){
                    center_of_objRect.y = half_warp.y+1;
                }
                if(center_of_objRect.y > (frame_Height - half_warp.y-1) ){
                    center_of_objRect.y = frame_Height - half_warp.y-1;
                }
            }
            cv::Point newBR = center_of_objRect + half_warp;
            cv::Point newTL = center_of_objRect - half_warp;
#ifdef DEBUG
            cout << "Center_of_ObjRect: " << center_of_objRect << endl;
#endif
            // adjust newBR and newTL
            if(newBR.x>frame_Width) newBR.x = frame_Width;
            if(newBR.y>frame_Height) newBR.y = frame_Height;
            if(newTL.x<0) newTL.x = 0;
            if(newTL.y<0) newTL.y = 0;
#ifdef DEBUG
            cout << "newTL: " << newTL << endl;
            cout << "newBR: " << newBR << endl;
            cout << "Frame [WxH] info: [" << frame_Width << "," << frame_Height << "]" << endl;
            cout << "Tracked roi width and height: " << newBR.x - newTL.x << ", " << newBR.y - newTL.y << endl;
            cout << "GIF roi width and height: " << gray_gifFrame.cols << ", " << gray_gifFrame.rows << endl;
#endif
            if ((newBR.y - newTL.y) != gray_gifFrame.rows){
                newBR.y += 1;
            }
            if((newBR.x - newTL.x) != gray_gifFrame.cols){
                newBR.x += 1;
            }
            Rect NewObjRect = Rect(newBR, newTL);
            Mat roi = frame(NewObjRect);

            Mat mask = im_warp_vector[gifID].clone();
            Mat mask_gray = imGray_warp_vector[gifID].clone();

            imshow("GIF_Mask", mask);
            imshow("ROI", roi);
            mask.copyTo(roi, mask_gray);
            
            // update previous status
            prev_prev_frame[i] = prev_frame[i];
            prev_frame[i] = current_frame;
        }
        frameID++;
        gifID++;
        gettimeofday(&end, NULL);
#ifdef DEBUG
        std::cout << "track cost time : " << 1000*1000*(end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec) << " us" << std::endl;
        std::cout << "##################" << endl;
#endif
        if(gifID>=imGray_warp_vector.size())
            gifID = 0;
        
        // show frame
        imshow("GIF_Tracking", frame);

#ifdef SAVE_VIDEO
        // imwrite("frames/"+ to_string(frameID)+".jpg", frame);
        writer << frame;
#endif
        // quit on x button
        if  (waitKey(1) == 27) break;
    }
    cap.release();
#ifdef SAVE_VIDEO
    writer.release();
#endif
    destroyAllWindows();
    return 0;
}
