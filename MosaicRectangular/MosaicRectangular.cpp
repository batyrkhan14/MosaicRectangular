#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<dirent.h>
#include<cstring>
#include<set>
#include<string>
#include<iostream>
#include<conio.h>           
#define INF 2000000000

using namespace cv;
using namespace std;

const string INPUT_IMAGE_PATH = "D:\\study\\MosaicImages\\image.jpeg";
const string DATASET_DIRECTORY = "D:\\study\\MosaicImages\\my_dataset\\";
const int PATCH_WIDTH = 40;
const int PATCH_HEIGHT = 40;


Mat inputImage, resultImage;
int height, width;
int windowsCounter = 0;
vector<vector<int> > minVals;
vector<vector<string> > patchNames;

void initialize() {
	inputImage = imread(INPUT_IMAGE_PATH);
	height = inputImage.rows - inputImage.rows%PATCH_HEIGHT;
	width = inputImage.cols - inputImage.cols%PATCH_WIDTH;
	resize(inputImage, inputImage, Size(width, height));
	resultImage = inputImage.clone();
	for (int i = 0; i < height; i += PATCH_HEIGHT) {
		vector<int> temp;
		vector<string> stemp;
		for (int j = 0; j < width; j += PATCH_WIDTH) {
			temp.push_back(INF);
			stemp.push_back("");
		}
		minVals.push_back(temp);
		patchNames.push_back(stemp);
	}
}

void showImage(Mat image) {
	string name = "Image" + to_string(++windowsCounter);
	namedWindow(name, WINDOW_AUTOSIZE);
	imshow(name, image);
	waitKey(0);
}

int diff(Mat image1, Mat image2) {
	int result = 0;
	int n = image1.cols;
	int m = image1.rows;
	for (int i = 0; i < image1.cols; i++) {
		for (int j = 0; j < image1.rows; j++) {
			Vec3b intensity1 = image1.at<Vec3b>(j, i);
			Vec3b intensity2 = image2.at<Vec3b>(j, i);
			for (int k = 0; k < 3; k++) {
				result += ((int)intensity1.val[k] - (int)intensity2.val[k])*((int)intensity1.val[k] - (int)intensity2.val[k]);
			}
		}
	}
	return result;
}

int diffMeans(Mat image1, Mat image2) {
	int result = 0;
	int n = image1.cols;
	int m = image1.rows;
	Scalar mean1 = mean(image1);
	Scalar mean2 = mean(image2);
	for (int i = 0; i < 3; i++) {
		result += ((int)mean1.val[i] - (int)mean2.val[i])*((int)mean1.val[i] - (int)mean2.val[i]);
	}
	return result;
}


void processImage(String imagePath) {
	cout << imagePath << endl;
	set<pair<int, int> > st;
	Mat image = imread(imagePath);
	resize(image, image, Size(PATCH_WIDTH, PATCH_HEIGHT));
	for (int i = 0, ii = 0; i < height; i += PATCH_HEIGHT, ii++) {
		for (int j = 0, jj = 0; j < width; j += PATCH_WIDTH, jj++) {
			Mat piece = inputImage(Rect(j, i, PATCH_WIDTH, PATCH_HEIGHT));
			//int df = diff(piece, image); // difference pixel by pixel - works slow
			int df = diffMeans(piece, image); // difference of means - works faster
			if (df < minVals[ii][jj] && (ii == 0 || st.find(make_pair(ii - 1, jj)) == st.end())
				&& (jj == 0 || st.find(make_pair(ii, jj - 1)) == st.end())
				&& (ii == (int)minVals.size() - 1 || st.find(make_pair(ii + 1, jj)) == st.end())
				&& (jj == (int)minVals[0].size() - 1 || st.find(make_pair(ii, jj + 1)) == st.end())) {
				st.insert(make_pair(ii, jj));
				minVals[ii][jj] = df;
				patchNames[ii][jj] = imagePath;
			}
		}
	}
}

void readDataImages(String dirName) {
	DIR* dir;
	dir = opendir(dirName.c_str());
	dirent* pdir;
	while (pdir = readdir(dir)) {
		if (pdir->d_type == DT_DIR) {
			if (strcmp(pdir->d_name, ".") != 0 && strcmp(pdir->d_name, "..")) {
				readDataImages(dirName + pdir->d_name + "\\");
			}
		}
		else {
			processImage(dirName + pdir->d_name);
		}
	}
}

void adjust() {
	cout << "Adjusting..." << endl;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			double coef = (((i - height / 2)*(i - height / 2) + (j - width / 2)*(j - width / 2) / 2)*1.0 / (width*width / 12) + 0.3) / 1.3;
			if (coef > 1.0) coef = 1.0;
			for (int k = 0; k < 3; k++) {
				resultImage.at<Vec3b>(i, j).val[k] = inputImage.at<Vec3b>(i, j).val[k] * (1.0 - coef) + resultImage.at<Vec3b>(i, j).val[k] * coef;
			}
		}
	}
}

void createResultImage() {
	cout << "Creating result image..." << endl;
	for (int i = 0, ii = 0; i < height; i += PATCH_HEIGHT, ii++) {
		for (int j = 0, jj = 0; j < width; j += PATCH_WIDTH, jj++) {
			Mat image = imread(patchNames[ii][jj]);
			resize(image, image, Size(PATCH_WIDTH, PATCH_HEIGHT));
			image.copyTo(resultImage(Rect(j, i, PATCH_WIDTH, PATCH_HEIGHT)));
		}
	}
}

int main() {
	initialize();
	readDataImages(DATASET_DIRECTORY);
	createResultImage();
	adjust();
	showImage(resultImage);
	imwrite("D:\\result.jpg", resultImage);
	_getch();
	return 0;
}

