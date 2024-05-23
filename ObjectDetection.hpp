//
// Created by lz on 26.05.23.
//

#ifndef HAR_OBJECTDETECTION_HPP
#define HAR_OBJECTDETECTION_HPP

#endif //HAR_OBJECTDETECTION_HPP

#include <yaml-cpp/yaml.h>
#include "yolo.h"
#include "BYTETracker.h"
#include <mutex>
#include <condition_variable>

using namespace std;
using namespace cv;
using namespace Eigen;
class ObjectDetector{
private:
    YAML::Node root;
    YAML::Node detect_config;
    YAML::Node tracker_config;
    YOLO yolo;
    BYTETracker tracker;
    bool yolo_only = false;
public:
    explicit ObjectDetector();
    ~ObjectDetector();
    void initDetector(const string& config_file);
    void run(const Mat&, const Mat& , const double&, const double&,
             const double&, const double&, const Vector3d&);
    void get3dObjs(const Mat&, const double&, const double&,
                   const double&, const double&, const Vector3d&);
    static double getDepthFromFrame(const Mat &frameD, float cx, float cy, int patch_size = 6);
    void display(const Mat &frame);
    vector<vector<DetectRes>> detect_boxes;
    vector<Box> Boxes_fi;
    MatrixXd Objs;
    vector<Vector3d> ObjsSameClass;
    bool succeed=false;
};

ObjectDetector::ObjectDetector() = default;

ObjectDetector::~ObjectDetector() = default;

void ObjectDetector::initDetector(const string& config_file){
    root = YAML::LoadFile(config_file);
    detect_config = root["yolo"];
    tracker_config = root["tracker"];
    yolo.initYOLO(detect_config);
    tracker.initTracker(tracker_config);
    yolo.LoadEngine();
    succeed = false;
}

void ObjectDetector::run(const Mat &frame, const Mat &frameD, const double &fx, const double &fy,
                         const double &cx, const double &cy, const Vector3d &reference){
    succeed = false;
    vector<cv::Mat> vec_org_img;
    vec_org_img.push_back(frame);
    detect_boxes = yolo.InferenceImages(vec_org_img);
    Boxes_fi.clear();
    Objs = MatrixXd::Zero(3,tracker.class_ids.size());
    if (!detect_boxes[0].empty()) {
        std::vector<STrack> output_stracks = tracker.update(detect_boxes[0]);
        for (unsigned long i = 0; i < output_stracks.size(); i++) {
            std::vector<float> tlwh = output_stracks[i].tlwh;
            Box box;
            box.x = tlwh[0] + tlwh[2]/2;
            box.y = tlwh[1] + tlwh[3]/2;
            box.w = tlwh[2];
            box.h = tlwh[3];
            box.id = output_stracks[i].track_id;
            Boxes_fi.push_back(box);
        }
        get3dObjs(frameD, fx, fy, cx, cy, reference);
        succeed = true;
    }
}

void ObjectDetector::get3dObjs(const Mat &frameD, const double &fx, const double &fy,
                               const double &cx, const double &cy, const Vector3d &reference) {
    condition_variable cv_;
    mutex mutex_;
    unique_lock<std::mutex> lock(mutex_);
    if(!yolo_only) {
        for (const auto id: tracker.class_ids) {
            ObjsSameClass.clear();
            vector<double> dist;
            for (const auto box: Boxes_fi) {
                if (box.id != id)
                    continue;
                Vector3d obj;
                auto depth = getDepthFromFrame(frameD, box.x, box.y);
                double X = (box.x - cx) / fx * depth;
                double Y = (box.y - cy) / fy * depth;
                obj << X, Y, depth;
                ObjsSameClass.push_back(obj);
            }
            //choose the nearst object
            if (ObjsSameClass.size() == 1)
                Objs.col(id) = ObjsSameClass[0];
            else if (ObjsSameClass.size() > 1) {
//            if (cv_.wait_for(lock, std::chrono::milliseconds(30), [&skReady]{ return skReady; })) {
                for (const auto &Obj: ObjsSameClass) {
                    dist.push_back(sqrt(pow(reference[0] - Obj[0], 2) + pow(reference[1] - Obj[1], 2) +
                                        pow(reference[2] - Obj[2], 2)));
                }
                auto min_ele = std::min_element(dist.begin(), dist.end());
                int index = distance(dist.begin(), min_ele);
                Objs.col(id) = ObjsSameClass[index];
//            }
            }
        }
    }
    else{
        for (const auto id: tracker.class_ids) {
            ObjsSameClass.clear();
            vector<double> dist;
            for (const auto box: detect_boxes[0]) {
                if (box.classes != id)
                    continue;
                Vector3d obj;
                auto depth = getDepthFromFrame(frameD, box.x, box.y);
                double X = (box.x - cx) / fx * depth;
                double Y = (box.y - cy) / fy * depth;
                obj << X, Y, depth;
                ObjsSameClass.push_back(obj);
            }
            //choose the nearst object
            if (ObjsSameClass.size() == 1)
                Objs.col(id) = ObjsSameClass[0];
            else if (ObjsSameClass.size() > 1) {
//            if (cv_.wait_for(lock, std::chrono::milliseconds(30), [&skReady]{ return skReady; })) {
                for (const auto &Obj: ObjsSameClass) {
                    dist.push_back(sqrt(pow(reference[0] - Obj[0], 2) + pow(reference[1] - Obj[1], 2) +
                                        pow(reference[2] - Obj[2], 2)));
                }
                auto min_ele = std::min_element(dist.begin(), dist.end());
                int index = distance(dist.begin(), min_ele);
                Objs.col(id) = ObjsSameClass[index];
//            }
            }
        }
    }
}

double ObjectDetector::getDepthFromFrame(const Mat &frameD, float x, float y, int patch_size){
    int half_patch_size = patch_size / 2;
    double depth_mean;
    cv::Rect roi(int(x)- half_patch_size + 1, int(y) - half_patch_size + 1, patch_size, patch_size);
    if (roi.x < 0)
        roi.x = 0;
    if (roi.x > frameD.cols)
        roi.x = frameD.cols;
    if (roi.y < 0)
        roi.y = 0;
    if (roi.y > frameD.rows)
        roi.y = frameD.rows;
    if (roi.x + roi.width > frameD.cols)
        roi.width = frameD.cols - roi.x;
    if (roi.y + roi.height > frameD.rows)
        roi.height = frameD.rows - roi.y;
    cv::Mat patch = frameD(roi).clone();
    cv::threshold(patch, patch, 0, 0, cv::THRESH_TOZERO);
    int num_nonzero = cv::countNonZero(patch);
    double depth_sum = cv::sum(patch)[0] / 1000.0;
    if (num_nonzero > 0){
        depth_mean = depth_sum / num_nonzero;
    }
    else{
        depth_mean = 0;
    }
    return depth_mean;
}

void ObjectDetector::display(const Mat &frame) {
    if(!yolo_only) {
        for (auto &Box: Boxes_fi) {
            //cout << Box.cx << endl;
            if (Box.w != 0) {
                auto point1 = cv::Point2f(Box.x - Box.w / 2, Box.y - Box.h / 2);
                auto point2 = cv::Point2f(Box.x + Box.w / 2, Box.y + Box.h / 2);
                cv::putText(frame, tracker.names[Box.id], cv::Point(Box.x - Box.w / 2, Box.y - Box.h / 2 - 5),
                            cv::FONT_HERSHEY_COMPLEX, 0.7, tracker.id_colors[Box.id], 2);
                cv::rectangle(frame, point1, point2, tracker.id_colors[Box.id], 2);
            }
        }
    }
    // resluts from yolo only
    else {
        if (detect_boxes[0].size() > 0) {
            for (auto &Box: detect_boxes[0]) {
                //cout << Box.cx << endl;
                if (Box.w != 0) {
                    auto point1 = cv::Point2f(Box.x - Box.w / 2, Box.y - Box.h / 2);
                    auto point2 = cv::Point2f(Box.x + Box.w / 2, Box.y + Box.h / 2);
                    cv::putText(frame, tracker.names[Box.classes] + " : " + to_string(int(Box.prob * 100)),
                                cv::Point(Box.x - Box.w / 2, Box.y - Box.h / 2 - 5),
                                cv::FONT_HERSHEY_COMPLEX, 0.7, tracker.id_colors[Box.classes], 2);
//                    cv::putText(frame, tracker.names[Box.classes],
//                                cv::Point(Box.x - Box.w / 2, Box.y - Box.h / 2 - 5),
//                                cv::FONT_HERSHEY_COMPLEX, 0.7, tracker.id_colors[Box.classes], 2);
                    cv::rectangle(frame, point1, point2, tracker.id_colors[Box.classes], 2);
                }
            }
        }
    }
}


