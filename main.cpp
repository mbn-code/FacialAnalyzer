#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <vector>
#include <fstream>   // For JSON file saving
#include <chrono>    // For waiting/delay if needed
#include <future>    // For async processing

using namespace cv;
using namespace std;

// Function to compute Intersection over Union (IoU) between two rectangles.
double computeIoU(const Rect& a, const Rect& b) {
    Rect inter = a & b;
    double interArea = static_cast<double>(inter.area());
    double unionArea = a.area() + b.area() - interArea;
    return unionArea > 0 ? interArea / unionArea : 0;
}

// Function to compute image smoothness via Laplacian variance (proxy for skin quality)
double computeSmoothness(const Mat &faceImg) {
    Mat gray, lap;
    cvtColor(faceImg, gray, COLOR_BGR2GRAY);
    Laplacian(gray, lap, CV_64F);
    Scalar mu, sigma;
    meanStdDev(lap, mu, sigma);
    return sigma.val[0] * sigma.val[0]; // variance: higher variance = more texture/noise
}

// Candidate struct to hold progressive results.
struct Candidate {
    Rect faceRect;
    double bestScore;
    string stats;
    bool active;
};

int main() {
    // Open the default camera.
    VideoCapture cap(1);
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
    string eyePath = cv::samples::findFile("haarcascades/haarcascade_eye.xml");
    if (!eyeCascade.load(eyePath))
        cerr << "Error loading eye cascade from: " << eyePath << endl;
    
    string mouthPath = cv::samples::findFile("haarcascades/haarcascade_smile.xml");
    if (!mouthCascade.load(mouthPath))
        cerr << "Error loading mouth cascade from: " << mouthPath << endl;

    // Vectors to store information from saved faces.
    vector<Rect> savedFaces;           // Saved bounding boxes.
    vector<Mat> savedHistograms;       // Histograms computed in HSV space.
    vector<Mat> savedORBDescriptors;   // ORB descriptors for each saved face.
    vector<string> savedStats;         // To store statistics for each person.

    // Candidate status.
    Candidate candidate;
    candidate.active = false;
    
    // Create an ORB feature extractor and BFMatcher.
    Ptr<ORB> orb = ORB::create();
    BFMatcher matcher(NORM_HAMMING);

    // Maximum number of saved faces.
    const size_t maxSavedFaces = 50;
    int personCounter = 0;  // Candidate ID

    Mat frame;
    while (true) {
        cap >> frame;  // Capture a frame.
        if (frame.empty())
            break;

        // Prepare the frame for DNN-based face detection.
        Mat blob = dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0));
        net.setInput(blob);
        auto detectionsFuture = std::async(std::launch::async, [&](){ return net.forward(); });
        Mat detections = detectionsFuture.get();

        // Loop over detections.
        for (int i = 0; i < detections.size[2]; i++) {
            float confidence = detections.ptr<float>(0)[i * 7 + 2];
            if (confidence > 0.7) {  // Confidence threshold.
                int x1 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 3] * frame.cols);
                int y1 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 4] * frame.rows);
                int x2 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 5] * frame.cols);
                int y2 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 6] * frame.rows);
                Rect faceRect(Point(x1, y1), Point(x2, y2));
                faceRect &= Rect(0, 0, frame.cols, frame.rows); // Ensure within frame.

                // Draw the detection rectangle for visualization.
                rectangle(frame, faceRect, Scalar(0, 255, 0), 2);

                // Extract and clone the face region.
                Mat faceImg = frame(faceRect).clone();

                // Convert to grayscale for cascades.
                Mat faceGray;
                cvtColor(faceImg, faceGray, COLOR_BGR2GRAY);

                // --- Additional Beauty Analysis ---
                // 1. Facial Thirds & Fifths (Proportions)
                int thirdH = faceRect.height / 3;
                int thirdW = faceRect.width / 3;
                // Draw vertical grid lines (facial fifths can be approximated similarly).
                line(frame, Point(faceRect.x + thirdW, faceRect.y), Point(faceRect.x + thirdW, faceRect.y + faceRect.height), Scalar(0, 0, 255), 1);
                line(frame, Point(faceRect.x + 2 * thirdW, faceRect.y), Point(faceRect.x + 2 * thirdW, faceRect.y + faceRect.height), Scalar(0, 0, 255), 1);
                // Draw horizontal grid lines (facial thirds).
                line(frame, Point(faceRect.x, faceRect.y + thirdH), Point(faceRect.x + faceRect.width, faceRect.y + thirdH), Scalar(0, 0, 255), 1);
                line(frame, Point(faceRect.x, faceRect.y + 2 * thirdH), Point(faceRect.x + faceRect.width, faceRect.y + 2 * thirdH), Scalar(0, 0, 255), 1);

                // 2. Eye Detection & Analysis (Size, Spacing, Symmetry)
                vector<Rect> eyes;
                eyeCascade.detectMultiScale(faceGray, eyes, 1.1, 4, 0, Size(30, 30));
                for (size_t e = 0; e < eyes.size(); e++) {
                    Rect eyeRect = eyes[e];
                    // Adjust coordinates relative to full frame.
                    eyeRect.x += faceRect.x;
                    eyeRect.y += faceRect.y;
                    rectangle(frame, eyeRect, Scalar(255, 0, 0), 2);
                }
                string eyeAnalysis = "Eyes: ";
                if (eyes.size() == 2) {
                    // Assume ideal if two eyes of similar size and spacing roughly equal to one eye-width apart.
                    int eyeSizeDiff = abs(eyes[0].width - eyes[1].width);
                    int eyeDistance = abs((eyes[0].x + eyes[0].width/2) - (eyes[1].x + eyes[1].width/2));
                    eyeAnalysis += "Detected two eyes. ";
                    eyeAnalysis += (eyeSizeDiff < 5) ? "Similar size; " : "Size difference; ";
                    eyeAnalysis += (eyeDistance > eyes[0].width * 0.8 && eyeDistance < eyes[0].width * 2.0) ? "Spacing acceptable." : "Spacing unusual.";
                } else {
                    eyeAnalysis += "Non-optimal eye detection.";
                }

                // 3. Mouth (Smile) Detection (as a proxy for lip/expression analysis)
                vector<Rect> mouths;
                Rect lowerHalf(0, faceGray.rows / 2, faceGray.cols, faceGray.rows / 2);
                Mat lowerGray = faceGray(lowerHalf);
                mouthCascade.detectMultiScale(lowerGray, mouths, 1.1, 2, 0, Size(30, 30));
                string mouthAnalysis = "Mouth: ";
                if (!mouths.empty()) {
                    mouthAnalysis += "Smile detected.";
                    // Draw mouth rectangle (adjust coordinates).
                    Rect mouthRect = mouths[0];
                    mouthRect.x += faceRect.x;
                    mouthRect.y += faceRect.y + faceGray.rows/2;
                    rectangle(frame, mouthRect, Scalar(0, 0, 255), 2);
                } else {
                    mouthAnalysis += "No smile detected.";
                }

                // 4. Skin Quality (Smoothness measure)
                double smoothness = computeSmoothness(faceImg);
                // In this example, a lower Laplacian variance suggests smoother skin.
                string skinAnalysis = "Skin: ";
                skinAnalysis += (smoothness < 100) ? "Smooth" : "Textured";

                // 5. Existing analysis: Brightness, ORB features, symmetry (brightness-based)
                double brightness = mean(faceImg)[0];
                double brightnessDiff = fabs(brightness - 120.0);
                double brightnessScore = max(0.0, 100.0 - brightnessDiff);

                // ORB features.
                vector<KeyPoint> keypoints;
                Mat descriptors;
                Mat candidateGray;
                cvtColor(faceImg, candidateGray, COLOR_BGR2GRAY);
                resize(candidateGray, candidateGray, Size(100, 100)); // Resize for consistency.
                orb->detectAndCompute(candidateGray, Mat(), keypoints, descriptors);

                int kpCount = keypoints.size();
                double kpDiff = fabs(kpCount - 50);
                double kpScore = max(0.0, 100.0 - kpDiff);

                // Compute symmetry using brightness difference between left and right halves.
                Rect leftHalf(faceRect.x, faceRect.y, faceRect.width/2, faceRect.height);
                Rect rightHalf(faceRect.x + faceRect.width/2, faceRect.y, faceRect.width - faceRect.width/2, faceRect.height);
                double leftMean = mean(frame(leftHalf))[0];
                double rightMean = mean(frame(rightHalf))[0];
                double symmetryDiff = fabs(leftMean - rightMean);
                double symmetryScore = max(0.0, 100.0 - symmetryDiff);
                string symmetryAnalysis = (symmetryDiff <= 10) ? "Highly symmetrical" :
                                          (symmetryDiff <= 30) ? "Moderately symmetrical" : "Poor symmetry";

                // 6. Combine analyses for an overall beauty score.
                // (You can weight each parameter as desired. Here we simply average several scores.)
                double overallScore = (brightnessScore + kpScore + symmetryScore + 100.0) / 4.0;
                // Adjust overall score based on additional beauty analyses.
                // For example, bonus if eyes and smile meet criteria.
                if (eyes.size() == 2)
                    overallScore += 5;
                if (!mouths.empty())
                    overallScore += 5;
                if (smoothness < 100)
                    overallScore += 5;
                overallScore = min(overallScore, 100.0);

                // Prepare detailed JSON result.
                ostringstream json;
                json << "{\n"
                     << "  \"id\": " << (personCounter+1) << ",\n"
                     << "  \"brightness\": " << brightness << ",\n"
                     << "  \"brightnessScore\": " << brightnessScore << ",\n"
                     << "  \"eyeAnalysis\": \"" << eyeAnalysis << "\",\n"
                     << "  \"mouthAnalysis\": \"" << mouthAnalysis << "\",\n"
                     << "  \"skinAnalysis\": \"" << skinAnalysis << "\",\n"
                     << "  \"symmetryScore\": " << symmetryScore << ",\n"
                     << "  \"symmetryAnalysis\": \"" << symmetryAnalysis << "\",\n"
                     << "  \"keypoints\": " << kpCount << ",\n"
                     << "  \"keypointsScore\": " << kpScore << ",\n"
                     << "  \"overallScore\": " << overallScore << ",\n"
                     << "  \"overallAnalysis\": \"" << (overallScore > 80 ? "Excellent facial features" : overallScore > 60 ? "Good facial features" : "Needs improvement") << "\"\n"
                     << "}";
                // Save JSON to file.
                string jsonFilename = "result_" + to_string(personCounter+1) + ".json";
                ofstream outFile(jsonFilename);
                if (outFile.is_open()) {
                    outFile << json.str();
                    outFile.close();
                }

                // Duplicate checking (as in your original code).
                bool duplicate = false;
                int duplicateIdx = -1;
                // IoU Check.
                for (size_t j = 0; j < savedFaces.size(); j++) {
                    if (computeIoU(faceRect, savedFaces[j]) > 0.5) {
                        duplicate = true;
                        duplicateIdx = j;
                        break;
                    }
                }
                // Histogram Comparison.
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
                // ORB Feature Matching.
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

                // Overlay stats on the detected face.
                ostringstream statsStream;
                statsStream << "ID:" << (duplicate ? duplicateIdx+1 : personCounter+1)
                            << " Score:" << (int)overallScore
                            << " KP:" << kpCount;
                // Append additional beauty details.
                statsStream << " | " << eyeAnalysis << " | " << mouthAnalysis << " | " << skinAnalysis;
                string stats = statsStream.str();

                if (duplicate) {
                    // Update saved face rectangle.
                    savedFaces[duplicateIdx] = faceRect;
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
                    savedStats.push_back(stats);
                    personCounter++;
                    // Save face image asynchronously.
                    string filename = "pictures/person_" + to_string(personCounter) +
                                      "_score" + to_string((int)overallScore) +
                                      "_bright" + to_string((int)brightness) +
                                      "_kp" + to_string(kpCount) + ".jpg";
                    thread([faceImg, filename]() { imwrite(filename, faceImg); }).detach();
                }

                // Candidate update logic.
                if (candidate.active && (computeIoU(faceRect, candidate.faceRect) > 0.5)) {
                    // If overlapping, update candidate info if score is higher.
                    if (overallScore > candidate.bestScore) {
                        candidate.bestScore = overallScore;
                        candidate.stats = stats;
                        candidate.faceRect = faceRect;
                    }
                } else if (!candidate.active) {
                    candidate.active = true;
                    candidate.bestScore = overallScore;
                    candidate.stats = stats;
                    candidate.faceRect = faceRect;
                }
                
                // Draw the candidate rectangle.
                rectangle(frame, candidate.faceRect, Scalar(255, 255, 0), 2);
                // Display candidate stats.
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

                // Finalize if the candidate meets a quality threshold.
                if (candidate.active && candidate.bestScore >= 80) {
                    // Check for sufficient details (using keypoints, brightness, contrast, and face area).
                    Mat grayFaceForData;
                    cvtColor(faceImg, grayFaceForData, COLOR_BGR2GRAY);
                    Scalar meanGray, stdDevGray;
                    meanStdDev(grayFaceForData, meanGray, stdDevGray);
                    double contrast = stdDevGray[0];
                    const int minKeypointsThreshold = 15; // lowered threshold
                    bool sufficientData = (kpCount >= minKeypointsThreshold) &&
                                          (brightness > 30) && 
                                          (contrast > 15) && 
                                          (faceRect.area() > 800);
                    
                    if (sufficientData) {
                        putText(frame, "Final Score Reached!", Point(10, frame.rows - 30),
                                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
                        imshow("Enhanced Face Recognition", frame);
                        waitKey(1500);
                        goto finish;
                    } else {
                        putText(frame, "Insufficient detail; please adjust face position", 
                                Point(10, frame.rows - 60), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
                    }
                }
            }
        }

        // Overlay continuously updating stats.
        int offsetY = 30;
        for (const auto& s : savedStats) {
            putText(frame, s, Point(10, offsetY), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);
            offsetY += 30;
        }

        // Optionally, overlay detection confidence.
        ostringstream ss;
        ss << fixed << setprecision(1) << (detections.ptr<float>(0)[2] * 100);
        putText(frame, "Confidence: " + ss.str(), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        imshow("Enhanced Face Recognition", frame);
        if (waitKey(1) == 'q')
            break;
    }
finish:
    cap.release();
    destroyAllWindows();
    return 0;
}

// Example thresholds (adjust these values as needed)
const int MIN_KEYPOINTS = 50; // Reduced from a higher value
const double MIN_BRIGHTNESS = 0.3; // Reduced from a higher value
const double MIN_CONTRAST = 0.3; // Reduced from a higher value
const double MIN_FACE_SIZE = 0.1; // Reduced from a higher value

bool sufficientData(int kpCount, double brightness, double contrast, double faceArea) {
    return kpCount >= MIN_KEYPOINTS &&
           brightness >= MIN_BRIGHTNESS &&
           contrast >= MIN_CONTRAST &&
           faceArea >= MIN_FACE_SIZE;
}
