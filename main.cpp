#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

cv::Mat frame;
cv::Rect roi;
cv::Point start = cv::Point(0, 0);
cv::Point end = cv::Point(0, 0);
std::string windowName = "Ventana";
bool clicked = false;
bool regionSelected = false;

void modifyROI(cv::Rect &roi)
{
  roi.width = abs(end.x - start.x);
  roi.height = abs(end.y - start.y);

  if (start.x < end.x)
    roi.x = start.x;
  else
    roi.x = end.x;

  if (start.y < end.y)
    roi.y = start.y;
  else
    roi.y = end.y;
}

void drawImage(const std::string &windowName, cv::Mat frame, cv::Rect &roi)
{
  cv::Mat modifiedFrame;

  frame.copyTo(modifiedFrame);

  cv::rectangle(modifiedFrame, roi, cv::Scalar(0, 0, 255), 1, CV_AA);

  cv::imshow(windowName, modifiedFrame);
}

void onMouse(int event, int x, int y, int, void *)
{
  switch (event)
  {
  case cv::EVENT_LBUTTONDOWN:
    start = cv::Point(x, y);
    end = cv::Point(x, y);
    clicked = true;
    break;
  case cv::EVENT_LBUTTONUP:
    end = cv::Point(x, y);

    if (start != end)
    {
      if ((start.x == end.x) || (start.y == end.y))
        regionSelected = false;
      else
      {
        modifyROI(roi);
        regionSelected = true;
      }
    }

    else
    {
      start = cv::Point(0, 0);
      end = cv::Point(0, 0);
    }
    break;
  case cv::EVENT_MOUSEMOVE:
    if (clicked)
    {
      end = cv::Point(x, y);
      modifyROI(roi);
      drawImage(windowName, frame, roi);
    }
    break;
  default:
    break;
  }
}

int main(int argc, char** argv)
{
  cv::VideoCapture vc(argv[1]);
  bool videoFinished = false;

  if (!vc.isOpened())  // check if we succeeded
    return -1;
  cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);

  vc >> frame;

  cv::setMouseCallback(windowName, onMouse, &roi);

  while (!regionSelected)
  {
    drawImage(windowName, frame, roi);
    cv::waitKey(5);
  }

  cv::setMouseCallback(windowName, NULL, NULL);

  cv::Mat roiMat = frame(roi);
  cv::Mat objectHist;

  int histSize = 128;
  float range[] = { 0, 256 };

  int channels[] = { 0, 1, 2 };
  int histSizes[] = { histSize, histSize, histSize };
  const float* histRanges[] = { range, range, range };

  cv::calcHist(&roiMat, 1, channels, cv::Mat(), objectHist, 1, histSizes, histRanges);
  cv::normalize(objectHist, objectHist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

  while (!videoFinished)
  {
    cv::Mat tmp;
    if (!videoFinished)
    {
      vc >> tmp;
      if (tmp.empty())
        videoFinished = true;
      else
        frame = tmp;
    }

    double bestDistance = 1;
    cv::Rect bestRoi;

    for (int i = 0; i < 200; ++i)
    {
      int x = roi.x;
      int y = roi.y;
      int delta;

      delta = rand() % 20 - 10;
      x += delta;
      if (x < 0)
        x = 0;
      if (x + roi.width > frame.cols)
        x = frame.cols - roi.width - 1;

      delta = rand() % 20 - 10;
      y += delta;
      if (y < 0)
        y = 0;
      if (y + roi.height > frame.rows)
        y = frame.rows - roi.height - 1;

      cv::Rect testRoiRect(x, y, roi.width, roi.height);
      cv::Mat testRoi = frame(testRoiRect);

      cv::Mat testRoiHist;
      cv::calcHist(&testRoi, 1, channels, cv::Mat(), testRoiHist, 1, histSizes, histRanges);
      cv::normalize(testRoiHist, testRoiHist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

      double distance = cv::compareHist(objectHist, testRoiHist, CV_COMP_BHATTACHARYYA);

      if (distance <= bestDistance) {
        bestDistance = distance;
        bestRoi = testRoiRect;
      }
    }

    roi = bestRoi;

    drawImage(windowName, frame, roi);
    cv::waitKey(25);
  }
}