#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\ocl\ocl.hpp>
#include<iostream>
#include<vector>
#include<cmath>
#include<Windows.h>

using namespace std;
using namespace cv;
using namespace cv::ocl;

int main()
{
	int iLowH = 1;
	int iHighH = 179;

	int iLowS = 0; 
	int iHighS = 255;

	int iLowV = 0;
	int iHighV = 255;

	//Create trackbars in "Control" window
	namedWindow("Central",CV_WINDOW_AUTOSIZE);
	cvCreateTrackbar("LowH", "Central", &iLowH, 179); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Central", &iHighH, 179);

	cvCreateTrackbar("LowS", "Central", &iLowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Central", &iHighS, 255);

	cvCreateTrackbar("LowV", "Central", &iLowV, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Central", &iHighV, 255);

	VideoCapture cap(0);
	Mat image,trajectory;
	namedWindow("video",CV_WINDOW_AUTOSIZE);
	char ch;
	int update_bg_model=-1;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	int largest_contour_index;
	double contour_area;
	Point2f tmp1(0,0),tmp2(0,0);
	Moments m;
	bool mouse = false;
	bool circ = true;
	bool gest = false;

	Mat fgimg,fgmask,img;

	cap>>image;
	trajectory=Mat::zeros(image.rows,image.cols,image.type());
	cv::ocl::MOG2 mog2;
	oclMat dimage(image);
	oclMat dfgimg,dfgmask,dimg;
	dfgimg.create(image.size(),image.type());

	int flag=0;
	while(true)
	{
		cap>>image;
		
		dimage.upload(image);
		mog2(dimage, dfgmask,update_bg_model);
        mog2.getBackgroundImage(dimg);

		dfgmask.download(fgmask);
        dfgimg.download(fgimg);
        if (!dimg.empty())
            dimg.download(img);

		erode(fgmask, fgmask, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );
		//dilate( backproj, backproj, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) ); 
		blur(fgmask,fgmask,Size(10,10));
		threshold(fgmask,fgmask,10,255,CV_THRESH_BINARY);
		fgimg.setTo(Scalar::all(0));
		image.copyTo(fgimg,fgmask);

		//cvtColor(fgimg,fgimg,CV_BGR2HSV);
		//inRange(fgimg, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), fgimg); //Threshold the image
		//fgimg.copyTo(fgmask);
		erode(fgmask, fgmask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
		
		findContours(fgmask,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
		vector<vector<int> >hull(1);
		vector<vector<Point> >hull_pt(1);
		vector<Vec4i> defects;

		if(!contours.empty())
		{
			largest_contour_index=0;
			contour_area=contours[0].size();
			for(int i=1;i<contours.size();i++)
			{
				if(contours[i].size()>contour_area)
				{
					largest_contour_index=i;
					contour_area=contours[i].size();
				}
			}
			if(contourArea(contours[largest_contour_index])>100)
			{
				m=moments(contours[largest_contour_index]);
				tmp1=tmp2;
				tmp2.x=640-m.m10/m.m00;
				tmp2.y=m.m01/m.m00;
				drawContours(image,contours,largest_contour_index,Scalar(255,0,0),2);
				circle(trajectory,tmp2,5,Scalar(0,0,255),2);
				

				if(flag==0)
				{
					flag=1;
					continue;
				}
				if(tmp1.y<tmp2.y)
				{
					line(trajectory,tmp1,tmp2,Scalar(0,255,0),2);
				}
				else line(trajectory,tmp1,tmp2,Scalar(255,0,0),2);

				// Convex Hull
				convexHull( Mat(contours[largest_contour_index]), hull[0], false );
				convexHull( Mat(contours[largest_contour_index]), hull_pt[0], false );
				convexityDefects(contours[largest_contour_index], hull[0], defects);
				drawContours( image, hull_pt, 0, Scalar(0,0,255), 2, 8, vector<Vec4i>(), 0, Point() );

				if(defects.size()>5)
				{
				vector<float> distances,fdistances;
				vector<Vec4i> index,findex;
				vector<Point2f> bounding_ellipse;
				for(int i=0;i<defects.size();i++)
				{
					index.push_back(defects[i]);
					distances.push_back(defects[i][3]);
				}

				for(int i=0;i<6;i++)
				{
					int ptr=distance(distances.begin(),max_element(distances.begin(),distances.end()));
					fdistances.push_back(distances[ptr]);
					findex.push_back(index[ptr]);
					bounding_ellipse.push_back(contours[largest_contour_index][index[ptr][2]]);
					circle(image,contours[largest_contour_index][index[ptr][0]],5,Scalar(0,255,0),2);
					circle(image,contours[largest_contour_index][index[ptr][1]],5,Scalar(0,255,0),2);
					circle(image,contours[largest_contour_index][index[ptr][2]],5,Scalar(0,255,0),2);

					distances.erase(distances.begin()+ptr);
					index.erase(index.begin()+ptr);
				}
				if(circ)
				{ellipse(image,fitEllipse(bounding_ellipse),Scalar(0,255,0),2,8);
				Point2f mouse_movement=fitEllipse(bounding_ellipse).center;
				circle(image,mouse_movement,3,Scalar(0,0,255),3);
				if(mouse)
					SetCursorPos(mouse_movement.x*2.5,mouse_movement.y*2.5);}
				if(gest)
				{
					int i;
					for(i=0;i<6;i++)
					{
						//cout<<fdistances[i]<<"\n";
						if(fdistances[i]<5000)
						{
							putText(image,"Close",Point(100,100),FONT_HERSHEY_SCRIPT_SIMPLEX,2,Scalar(100,100,100),2);
							break;
						}
					}
					if(i==6)
						putText(image,"Open",Point(100,100),FONT_HERSHEY_SCRIPT_SIMPLEX,2,Scalar(100,100,100),2);
				}

				}
			}
		}
		
		contours.clear();
		hierarchy.clear();
		cv::imshow("video",image);
		cv::imshow("trajectory",trajectory);
		cv::imshow("foreground image", fgimg);

		if(!img.empty())
          cv::imshow("mean background image", img );

		ch=waitKey(1);

		switch (ch)
		{
		case 'c':
			trajectory.setTo(Scalar::all(0));
			circ=!circ;
			flag=0;
			break;
		case 's':
			update_bg_model=0;
			break;
		case 'r':
			update_bg_model=-1;
			break;
		case 'q':
			return 0;
			break;
		case 'm':
			mouse=!mouse;
			break;
		case 'g':
			gest=!gest;
			break;
		default:
			break;
		}
	}

	return 0;
}