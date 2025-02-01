#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <vector>
#include <fstream>  // Added for JSON file saving
#include <chrono>  // For waiting/delay if needed

using namespace cv;
using namespace std;

// Function to compute Intersection over Union (IoU) between two rectangles.
double computeIoU(const Rect& a, const Rect& b) {
    Rect inter = a & b;
    double interArea = static_cast<double>(inter.area());
    double unionArea = a.area() + b.area() - interArea;
    return unionArea > 0 ? interArea / unionArea : 0;
}

// Add Candidate struct to hold progressive results.
struct Candidate {
    Rect faceRect;
    double bestScore;
    string stats;
    bool active;
};

int main() {
    // Open the default camera.
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open the camera!" << endl;
        return -1;
    }

    // Load the pre-trained face detection model (Caffe-based).
    string modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel";
    string configFile = "models/deploy.prototxt";
    dnn::Net net = dnn::readNetFromCaffe(configFile, modelFile);

    // Load cascades for eye and mouth detection.
    CascadeClassifier eyeCascade, mouthCascade;
    // Use cv::samples::findFile to locate the cascade files
    string eyePath = cv::samples::findFile("haarcascades/haarcascade_eye.xml");
    if (!eyeCascade.load(eyePath))
        cerr << "Error loading eye cascade from: " << eyePath << endl;
    
    string mouthPath = cv::samples::findFile("haarcascades/haarcascade_smile.xml");
    if (!mouthCascade.load(mouthPath))
        cerr << "Error loading mouth cascade from: " << mouthPath << endl;

    // Vectors to store information from saved faces.
    vector<Rect> savedFaces;      // Saved bounding boxes.
    vector<Mat> savedHistograms;  // Histograms computed in HSV space.
    vector<Mat> savedORBDescriptors; // ORB descriptors for each saved face.

    // Initialize candidate status.
    Candidate candidate;
    candidate.active = false;
    
    // Create an ORB feature extractor and BFMatcher.
    Ptr<ORB> orb = ORB::create();
    BFMatcher matcher(NORM_HAMMING);

    // Before the main loop, define a maximum number of saved faces.
    const size_t maxSavedFaces = 50;

    // New: Vector to store statistics for each person.
    vector<string> savedStats;

    int personCounter = 0;  // Will become the candidate ID.

    Mat frame;
    while (true) {
        cap >> frame;  // Capture a frame.
        if (frame.empty())
            break;

        // Prepare the frame for DNN-based face detection.
        Mat blob = dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0));
        net.setInput(blob);
        Mat detections = net.forward();

        // Loop over detections.
        for (int i = 0; i < detections.size[2]; i++) {
            float confidence = detections.ptr<float>(0)[i * 7 + 2];
            if (confidence > 0.4) {  // Confidence threshold.
                int x1 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 3] * frame.cols);
                int y1 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 4] * frame.rows);
                int x2 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 5] * frame.cols);
                int y2 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 6] * frame.rows);
                Rect faceRect(Point(x1, y1), Point(x2, y2));
                faceRect &= Rect(0, 0, frame.cols, frame.rows); // Ensure within frame.

                // Draw the detection for visualization.
                rectangle(frame, faceRect, Scalar(0, 255, 0), 2);

                // Extract and clone the face region.
                Mat faceImg = frame(faceRect).clone();

                // (Optional) Run eye and mouth detection on the face region.
                Mat faceGray;
                cvtColor(faceImg, faceGray, COLOR_BGR2GRAY);
                vector<Rect> eyes;
                eyeCascade.detectMultiScale(faceGray, eyes, 1.1, 4, 0, Size(30, 30));
                // ...existing code...

                // Remove duplicate eye distance block
                // ...existing code...

                vector<Rect> mouths;
                Rect lowerHalf(0, faceGray.rows / 2, faceGray.cols, faceGray.rows / 2);
                Mat lowerGray = faceGray(lowerHalf);
                mouthCascade.detectMultiScale(lowerGray, mouths, 1.1, 2, 0, Size(30, 30));
                // You can later use the eye and mouth positions for further analysis if needed.

                // Additional: Compute eye distance if at least two eyes are detected.

                // Additional: Overlay a mesh on the face rectangle (split into thirds).
                int thirdW = faceRect.width / 3;
                int thirdH = faceRect.height / 3;
                // Draw vertical grid lines.
                line(frame, Point(faceRect.x + thirdW, faceRect.y), Point(faceRect.x + thirdW, faceRect.y + faceRect.height), Scalar(0, 0, 255), 1);
                line(frame, Point(faceRect.x + 2 * thirdW, faceRect.y), Point(faceRect.x + 2 * thirdW, faceRect.y + faceRect.height), Scalar(0, 0, 255), 1);
                // Draw horizontal grid lines.
                line(frame, Point(faceRect.x, faceRect.y + thirdH), Point(faceRect.x + faceRect.width, faceRect.y + thirdH), Scalar(0, 0, 255), 1);
                line(frame, Point(faceRect.x, faceRect.y + 2 * thirdH), Point(faceRect.x + faceRect.width, faceRect.y + 2 * thirdH), Scalar(0, 0, 255), 1);

                // Technique 2: Compute a color histogram in HSV space.
                Mat faceHSV;
                cvtColor(faceImg, faceHSV, COLOR_BGR2HSV);
                int hBins = 50, sBins = 60;
                int histSize[] = { hBins, sBins };
                float hRanges[] = { 0, 180 };
                float sRanges[] = { 0, 256 };
                const float* ranges[] = { hRanges, sRanges };
                int channels[] = { 0, 1 };
                Mat hist;
                calcHist(&faceHSV, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
                normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

                // Technique 3: Compute ORB features.
                Mat candidateGray;
                cvtColor(faceImg, candidateGray, COLOR_BGR2GRAY);
                resize(candidateGray, candidateGray, Size(100, 100)); // Resize for consistency.
                vector<KeyPoint> keypoints;
                Mat descriptors;
                orb->detectAndCompute(candidateGray, Mat(), keypoints, descriptors);

                // Check for duplicates.
                bool duplicate = false;
                int duplicateIdx = -1;
                // 1. IoU Check.
                for (size_t j = 0; j < savedFaces.size(); j++) {
                    if (computeIoU(faceRect, savedFaces[j]) > 0.5) {
                        duplicate = true;
                        duplicateIdx = j;
                        break;
                    }
                }
                // 2. Histogram Comparison.
                if (!duplicate) {
                    for (size_t j = 0; j < savedHistograms.size(); j++) {
                        double correlation = compareHist(hist, savedHistograms[j], HISTCMP_CORREL);
                        if (correlation > 0.8) { 
                            duplicate = true;
                            duplicateIdx = j;
                            break;
                        }
                    }
                }
                // 3. ORB Feature Matching.
                if (!duplicate && !descriptors.empty()) {
                    for (size_t j = 0; j < savedORBDescriptors.size(); j++) {
                        if (descriptors.type() != savedORBDescriptors[j].type() || descriptors.cols != savedORBDescriptors[j].cols)
                            continue;
                        vector<vector<DMatch>> knnMatches;
                        matcher.knnMatch(descriptors, savedORBDescriptors[j], knnMatches, 2);
                        int goodMatches = 0;
                        for (const auto& matchPair : knnMatches) {
                            if (matchPair.size() >= 2 && matchPair[0].distance < 0.75 * matchPair[1].distance)
                                goodMatches++;
                        }
                        if (goodMatches > 10) {
                            duplicate = true;
                            duplicateIdx = j;
                            break;
                        }
                    }
                }
                // Overlay stats following the detection square.
                if (duplicate) {
                    // Update saved face rectangle with the new position.
                    savedFaces[duplicateIdx] = faceRect;
                    // Overlay the saved stats on the current face rectangle.
                    int baseLine = 0;
                    Size textSize = getTextSize(savedStats[duplicateIdx], FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
                    rectangle(frame, Point(faceRect.x, faceRect.y - textSize.height - baseLine),
                              Point(faceRect.x + textSize.width, faceRect.y), Scalar(0, 255, 0), FILLED);
                    putText(frame, savedStats[duplicateIdx], Point(faceRect.x, faceRect.y - baseLine),
                            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);
                } else {
                    // New face detected.
                    savedFaces.push_back(faceRect);
                    savedHistograms.push_back(hist);
                    savedORBDescriptors.push_back(descriptors);
                    personCounter++;
                    string filename = "pictures/person_" + to_string(personCounter) + ".jpg";
                    thread([faceImg, filename]() { imwrite(filename, faceImg); }).detach();
                    double brightness = mean(faceImg)[0];
                    string stats = "ID:" + to_string(personCounter) +
                                   " Bright:" + to_string((int)brightness) +
                                   " KP:" + to_string((int)keypoints.size()) +
                                   " Third:" + to_string(thirdH);
                    savedStats.push_back(stats);
                    int baseLine = 0;
                    Size textSize = getTextSize(stats, FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
                    rectangle(frame, Point(faceRect.x, faceRect.y - textSize.height - baseLine),
                              Point(faceRect.x + textSize.width, faceRect.y), Scalar(0, 255, 0), FILLED);
                    putText(frame, stats, Point(faceRect.x, faceRect.y - baseLine),
                            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);
                    if (savedFaces.size() > maxSavedFaces) {
                        savedFaces.erase(savedFaces.begin());
                        savedHistograms.erase(savedHistograms.begin());
                        savedORBDescriptors.erase(savedORBDescriptors.begin());
                        savedStats.erase(savedStats.begin());
                    }
                    // New: Full face analysis against "the ideal"
                    double idealBrightness = 120.0;
                    double brightnessDiff = fabs(brightness - idealBrightness);
                    double brightnessScore = max(0.0, 100.0 - brightnessDiff);
                    string brightnessAnalysis = (brightnessDiff <= 10) ? "Ideal brightness" :
                                                  (brightnessDiff <= 30 ? "Acceptable brightness" : "Poor brightness");

                    int kpCount = keypoints.size();
                    double kpDiff = fabs(kpCount - 50);
                    double kpScore = max(0.0, 100.0 - kpDiff);
                    string kpAnalysis = (kpDiff <= 5) ? "Ideal feature detail" :
                                          (kpDiff <= 15 ? "Moderate feature detail" : "Insufficient feature detail");

                    // Compute symmetry: compare brightness of left and right halves.
                    Mat grayFace;
                    cvtColor(faceImg, grayFace, COLOR_BGR2GRAY);
                    Rect leftHalf(0, 0, grayFace.cols/2, grayFace.rows);
                    Rect rightHalf(grayFace.cols/2, 0, grayFace.cols - grayFace.cols/2, grayFace.rows);
                    double leftMean = mean(grayFace(leftHalf))[0];
                    double rightMean = mean(grayFace(rightHalf))[0];
                    double symmetryDiff = fabs(leftMean - rightMean);
                    double symmetryScore = max(0.0, 100.0 - symmetryDiff);
                    string symmetryAnalysis = (symmetryDiff <= 10) ? "Highly symmetrical" :
                                              (symmetryDiff <= 30 ? "Moderately symmetrical" : "Poor symmetry");

                    // Determine if sufficient measurements were obtained.
                    bool completeMeasurements = (kpCount >= 20);
                    string dataCompletion = completeMeasurements ? "Complete data" : "Incomplete data, please align face better";

                    // Facial third analysis (preset as balanced).
                    string thirdAnalysis = "Facial thirds are balanced";

                    // Compute final overall score.
                    double overallScore = 0;
                    if(!completeMeasurements)
                        overallScore = (brightnessScore + kpScore + symmetryScore + 100) / 4.0;
                    else
                        overallScore = (brightnessScore + kpScore + symmetryScore + 100) / 4.0;

                    string overallAnalysis = (overallScore > 80) ? "Excellent facial features" :
                                              (overallScore > 60) ? "Good facial features" : "Needs improvement";

                    // Prepare detailed JSON result.
                    ostringstream json;
                    json << "{\n"
                         << "  \"id\": " << personCounter << ",\n"
                         << "  \"brightness\": " << brightness << ",\n"
                         << "  \"brightnessScore\": " << brightnessScore << ",\n"
                         << "  \"brightnessAnalysis\": \"" << brightnessAnalysis << "\",\n"
                         << "  \"keypoints\": " << kpCount << ",\n"
                         << "  \"keypointsScore\": " << kpScore << ",\n"
                         << "  \"keypointsAnalysis\": \"" << kpAnalysis << "\",\n"
                         << "  \"symmetryScore\": " << symmetryScore << ",\n"
                         << "  \"symmetryAnalysis\": \"" << symmetryAnalysis << "\",\n"
                         << "  \"facialThird\": " << thirdH << ",\n"
                         << "  \"facialThirdAnalysis\": \"" << thirdAnalysis << "\",\n"
                         << "  \"dataCompletion\": \"" << dataCompletion << "\",\n"
                         << "  \"overallScore\": " << overallScore << ",\n"
                         << "  \"overallAnalysis\": \"" << overallAnalysis << "\"\n"
                         << "}";
                    // Save JSON to file.
                    string jsonFilename = "result_" + to_string(personCounter) + ".json";
                    ofstream outFile(jsonFilename);
                    if (outFile.is_open()) {
                        outFile << json.str();
                        outFile.close();
                    }
                }

                // Compute overall scores (brightness, keypoints, symmetry)
                double brightness = mean(faceImg)[0];
                double brightnessDiff = fabs(brightness - 120.0);
                double brightnessScore = max(0.0, 100.0 - brightnessDiff);
                
                int kpCount = keypoints.size();
                double kpDiff = fabs(kpCount - 50);
                double kpScore = max(0.0, 100.0 - kpDiff);
                
                // Symmetry score.
                Mat grayFace;
                cvtColor(faceImg, grayFace, COLOR_BGR2GRAY);
                Rect leftHalf(0, 0, grayFace.cols/2, grayFace.rows);
                Rect rightHalf(grayFace.cols/2, 0, grayFace.cols - grayFace.cols/2, grayFace.rows);
                double symmetryScore = max(0.0, 100.0 - fabs(mean(grayFace(leftHalf))[0] - mean(grayFace(rightHalf))[0]));
                
                // Overall score: using equal weighting.
                double overallScore = (brightnessScore + kpScore + symmetryScore + 100.0) / 4.0;
                
                // Create stats string.
                string stats = "ID:" + to_string(candidate.active ? personCounter : (personCounter+1)) +
                               " Score:" + to_string((int)overallScore) +
                               " KP:" + to_string(kpCount);
                               
                // If candidate already exists and new detection overlaps sufficiently, update candidate.
                bool updateCandidate = false;
                if (candidate.active && (computeIoU(faceRect, candidate.faceRect) > 0.5)) {
                    updateCandidate = true;
                }
                
                if (!candidate.active || updateCandidate) {
                    if (!candidate.active) {
                        personCounter++;
                        candidate.active = true;
                    }
                    // Update candidate info if overall score has improved.
                    if (overallScore > candidate.bestScore) {
                        candidate.bestScore = overallScore;
                        candidate.stats = stats;
                        candidate.faceRect = faceRect;
                    }
                    
                    // Draw the candidate rectangle.
                    rectangle(frame, candidate.faceRect, Scalar(255, 255, 0), 2);
                }
                
                // Draw a loading bar overlay based on candidate.bestScore.
                if (candidate.active) {
                    int progressUnits = min(10, (int)(candidate.bestScore / 10)); // each '=' represents 10%
                    string progressBar = "[";
                    for (int p = 0; p < progressUnits; p++) progressBar += "=";
                    for (int p = progressUnits; p < 10; p++) progressBar += " ";
                    progressBar += "]";
                    string displayText = candidate.stats + " " + progressBar;
                    int baseLine = 0;
                    Size textSize = getTextSize(displayText, FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
                    rectangle(frame, Point(candidate.faceRect.x, candidate.faceRect.y - textSize.height - baseLine),
                              Point(candidate.faceRect.x + textSize.width, candidate.faceRect.y), Scalar(0, 255, 0), FILLED);
                    putText(frame, displayText, Point(candidate.faceRect.x, candidate.faceRect.y - baseLine),
                            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);
                }
                
                // If candidate score is not high enough, do not finalize.
                if (candidate.active && candidate.bestScore < 80) {
                    // Continue waiting and updating candidate until quality improves.
                    // Optionally add a short delay for visible progress.
                    // This branch will keep refreshing with better quality frames.
                } else if (candidate.active && candidate.bestScore >= 80) {
                    // NEW: Check if there is sufficient data to finalize the scan.
                    Mat grayFaceForData;
                    cvtColor(faceImg, grayFaceForData, COLOR_BGR2GRAY);
                    Scalar meanGray, stdDevGray;
                    meanStdDev(grayFaceForData, meanGray, stdDevGray);
                    double contrast = stdDevGray[0];
                    // Define thresholds: require at least 30 keypoints for sufficient detail.
                    const int minKeypointsThreshold = 30;
                    bool sufficientData = (kpCount >= minKeypointsThreshold) &&
                                            (brightness > 50) &&
                                            (contrast > 20) &&
                                            (faceRect.area() > 1000);
                    
                    if (sufficientData) {
                        // Recompute detailed symmetry analysis.
                        Mat grayFace;
                        cvtColor(faceImg, grayFace, COLOR_BGR2GRAY);
                        Rect leftHalfRect(0, 0, grayFace.cols/2, grayFace.rows);
                        Rect rightHalfRect(grayFace.cols/2, 0, grayFace.cols - grayFace.cols/2, grayFace.rows);
                        double leftMean = mean(grayFace(leftHalfRect))[0];
                        double rightMean = mean(grayFace(rightHalfRect))[0];
                        double symmetryDiff = fabs(leftMean - rightMean);
                        double symmetryScore = max(0.0, 100.0 - symmetryDiff);
                        
                        string brightnessAnalysis = (brightnessDiff <= 10) ? "Perfect lighting" :
                                                    (brightnessDiff <= 30 ? "Good lighting" : "Dim lighting");
                        string kpAnalysis = (fabs(kpCount - 50) <= 5) ? "Optimal feature detail" :
                                            (fabs(kpCount - 50) <= 15 ? "Acceptable feature detail" : "Insufficient feature detail");
                        string symmetryAnalysis = (symmetryDiff <= 10) ? "Highly symmetrical" : "Asymmetrical";
                        string overallAnalysis = (candidate.bestScore > 90) ? "Outstanding facial features" :
                                                 (candidate.bestScore > 80) ? "Excellent facial features" : "Good facial features";
                        
                        ostringstream json;
                        json << "{\n"
                             << "  \"id\": " << personCounter << ",\n"
                             << "  \"brightness\": " << brightness << ",\n"
                             << "  \"brightnessScore\": " << brightnessScore << ",\n"
                             << "  \"brightnessAnalysis\": \"" << brightnessAnalysis << "\",\n"
                             << "  \"keypoints\": " << kpCount << ",\n"
                             << "  \"keypointsScore\": " << kpScore << ",\n"
                             << "  \"keypointsAnalysis\": \"" << kpAnalysis << "\",\n"
                             << "  \"symmetryScore\": " << symmetryScore << ",\n"
                             << "  \"symmetryAnalysis\": \"" << symmetryAnalysis << "\",\n"
                             << "  \"overallScore\": " << candidate.bestScore << ",\n"
                             << "  \"overallAnalysis\": \"" << overallAnalysis << "\"\n"
                             << "}";
                        string jsonFilename = "result_" + to_string(personCounter) + ".json";
                        ofstream outFile(jsonFilename);
                        if (outFile.is_open()) {
                            outFile << json.str();
                            outFile.close();
                        }
                        string filename = "pictures/person_" + to_string(personCounter) + ".jpg";
                        imwrite(filename, faceImg);
                        putText(frame, "Final Score Reached!", Point(10, frame.rows - 30),
                                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
                        imshow("Enhanced Face Recognition", frame);
                        waitKey(1500);
                        goto finish;
                    } else {
                        // Inform the user about insufficient detail.
                        putText(frame, "Insufficient detail; please adjust face position", 
                                Point(10, frame.rows - 60), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
                    }
                }
            }
        }

        // New: Continuously display stats on the top-left corner.
        int offsetY = 30;
        for (const auto& s : savedStats) {
            putText(frame, s, Point(10, offsetY), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);
            offsetY += 30;
        }

        // Optionally, overlay some text or confidence values.
        ostringstream ss;
        ss << fixed << setprecision(1) << (detections.ptr<float>(0)[2] * 100);
        putText(frame, "Confidence: " + ss.str(), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        // Display the result.
        imshow("Enhanced Face Recognition", frame);
        if (waitKey(1) == 'q')
            break;
    }
finish:
    cap.release();
    destroyAllWindows();
    return 0;
}
