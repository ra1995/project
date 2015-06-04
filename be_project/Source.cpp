#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2\core\core.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\ocl\ocl.hpp>
#include <opencv2\ml\ml.hpp>

#include <Windows.h>

#define OPEN 0
#define CLOSE 1

using namespace std;
using namespace cv;
using namespace cv::ocl;

class gesture
{
	vector<Vec4i> defects;
	vector<Point> contour;
	Point2f center;
	int fingers;
public:
	int flag;
	gesture(Point2f cent,vector<Vec4i> index,vector<Point> cont)
	{
		center=cent;
		defects=index;
		contour=cont;
		fingers=0;

		if(defects.size()<4)
			flag=CLOSE;
		else flag=OPEN;
	}

	void gesture_main()
	{
		double mean_defect_distance=0;

		for(int i=0;i<defects.size();i++)
		{
			mean_defect_distance+=distance(contour[defects[i][2]],center);
		}
		mean_defect_distance/=defects.size();
		
		for(int i=0;i<defects.size();i++)
		{
			if(i==0)
				if(distance(contour[defects[0][0]],center)>(2*mean_defect_distance))
				{
					fingers++;
				}
			if(distance(contour[defects[i][1]],center)>(2*mean_defect_distance))
			{
				fingers++;
			}
		}

		cout<<fingers<<"\n";
	}

	double distance(Point2f pt1,Point2f pt2)
	{
		double d=sqrt(pow((pt1.x-pt2.x),2)+pow((pt1.y-pt2.y),2));
		return d;
	}
};

class mouse
{
	int x,y;
	INPUT *buffer;
	bool mouse_hold_trigger;
public:
	int flag;
	mouse()
	{
		POINT mypoint;
		GetCursorPos(&mypoint);
		x=mypoint.x;
		y=mypoint.y;
		mouse_hold_trigger=false;
		flag=0;

		buffer=new INPUT;
		buffer->type=INPUT_MOUSE;
		buffer->mi.dx=(x*(0xFFFF/1366));
		buffer->mi.dy=(y*(0xFFFF/768));
		buffer->mi.mouseData=0;
		buffer->mi.dwFlags=MOUSEEVENTF_ABSOLUTE;
		buffer->mi.time=0;
		buffer->mi.dwExtraInfo=0;
	}
	void getmousepos()
	{
		POINT mypoint;
		GetCursorPos(&mypoint);
		x=mypoint.x;
		y=mypoint.y;
	}
	void updatemousepos(int mx,int my)
	{
		getmousepos();
		SetCursorPos(x+mx,y+my);
	}
	void MouseSetup()
	{
		buffer->mi.dx=(x*(0xFFFF/1366));
		buffer->mi.dy=(y*(0xFFFF/768));
	}

	void MouseLeftClick()
	{
		MouseSetup();
		buffer->mi.dwFlags=(MOUSEEVENTF_ABSOLUTE|MOUSEEVENTF_LEFTDOWN);
		SendInput(1,buffer,sizeof(INPUT));
		Sleep(10);
		buffer->mi.dwFlags=(MOUSEEVENTF_ABSOLUTE|MOUSEEVENTF_LEFTUP);
		SendInput(1,buffer,sizeof(INPUT));
		flag=1;
	}
	void MouseRightClick()
	{
		MouseSetup();
		buffer->mi.dwFlags=(MOUSEEVENTF_ABSOLUTE|MOUSEEVENTF_RIGHTDOWN);
		SendInput(1,buffer,sizeof(INPUT));
		Sleep(10);
		buffer->mi.dwFlags=(MOUSEEVENTF_ABSOLUTE|MOUSEEVENTF_RIGHTUP);
		SendInput(1,buffer,sizeof(INPUT));
		flag=1;
	}
	void MouseDoubleClick()
	{
		MouseSetup();
		buffer->mi.dwFlags=(MOUSEEVENTF_ABSOLUTE|MOUSEEVENTF_LEFTDOWN);
		SendInput(1,buffer,sizeof(INPUT));
		Sleep(10);
		buffer->mi.dwFlags=(MOUSEEVENTF_ABSOLUTE|MOUSEEVENTF_LEFTUP);
		SendInput(1,buffer,sizeof(INPUT));
		Sleep(10);
		buffer->mi.dwFlags=(MOUSEEVENTF_ABSOLUTE|MOUSEEVENTF_LEFTDOWN);
		SendInput(1,buffer,sizeof(INPUT));
		Sleep(10);
		buffer->mi.dwFlags=(MOUSEEVENTF_ABSOLUTE|MOUSEEVENTF_LEFTUP);
		SendInput(1,buffer,sizeof(INPUT));
		flag=1;
	}

	void mouse_hold_toggle()
	{
		mouse_hold_trigger=!mouse_hold_trigger;

		if(mouse_hold_trigger)
			mouse_hold_on();
		else mouse_hold_off();

		flag=1;
	}
	void mouse_hold_on()
	{
		MouseSetup();
		buffer->mi.dwFlags=(MOUSEEVENTF_ABSOLUTE|MOUSEEVENTF_LEFTDOWN);
		SendInput(1,buffer,sizeof(INPUT));
	}
	void mouse_hold_off()
	{
		buffer->mi.dwFlags=(MOUSEEVENTF_ABSOLUTE|MOUSEEVENTF_LEFTUP);
		SendInput(1,buffer,sizeof(INPUT));
	}
};

int main()
{
	VideoCapture cap(0);
	namedWindow("Video",CV_WINDOW_AUTOSIZE);
	namedWindow("Control",CV_WINDOW_AUTOSIZE);
	namedWindow("Mask",CV_WINDOW_AUTOSIZE);

	int hmin=0,hmax=180;
	int smin=0,smax=255;
	int vmin=0,vmax=255;
	createTrackbar("Hue Min","Control",&hmin,180,0);
	createTrackbar("Hue Max","Control",&hmax,180,0);
	createTrackbar("Sat Min","Control",&smin,255,0);
	createTrackbar("Sat Max","Control",&smax,255,0);
	createTrackbar("Val Min","Control",&vmin,255,0);
	createTrackbar("Val Max","Control",&vmax,255,0);

	Mat frame,frame_hsv,mask,img;
	oclMat oclmask;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<double> contour_area;
	Moments m;
	Point2f center;
	int display=0;
	int largest_contour_index;

	gesture *gesture_obj;
	mouse mouse_obj;

	while(true)
	{
		cap>>frame;
		if(frame.empty())
			continue;
		
		//convert color space
		cvtColor(frame,frame_hsv,CV_BGR2HSV);

		//create mask
		inRange(frame_hsv,Scalar(hmin,smin,vmin),Scalar(hmax,smax,vmax),mask);
		oclmask.upload(mask);
		ocl::erode(oclmask,oclmask,getStructuringElement(MORPH_ELLIPSE,Size(5,5),Point(2,2)));
		ocl::dilate(oclmask,oclmask,getStructuringElement(MORPH_ELLIPSE,Size(5,5),Point(2,2)));
		oclmask.download(mask);

		//apply mask
		img=Mat::zeros(frame_hsv.rows,frame_hsv.cols,frame_hsv.type());
		frame_hsv.copyTo(img,mask);

		//find contours
		Mat maskcpy=mask.clone();
		findContours(maskcpy,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
		vector<vector<int> >hull(1);
		vector<vector<Point> >hull_pt(1);
		vector<Vec4i> defects;

		if(!contours.empty())
		{
			//find largest contour
			for(int i=0;i<contours.size();i++)
			{
				contour_area.push_back(contourArea(contours[i]));
			}
			largest_contour_index=distance(contour_area.begin(),max_element(contour_area.begin(),contour_area.end()));

			if(contour_area[largest_contour_index]>1000)
			{
				drawContours(frame,contours,largest_contour_index,Scalar(255,0,0),2);

				//find center of contour
				m=moments(contours[largest_contour_index]);
				center.x=m.m10/m.m00;
				center.y=m.m01/m.m00;
				circle(frame,center,5,Scalar(0,255,0),2);

				//find convex hull
				convexHull( Mat(contours[largest_contour_index]), hull[0], false );
				convexHull( Mat(contours[largest_contour_index]), hull_pt[0], false );
				convexityDefects(contours[largest_contour_index], hull[0], defects);
				drawContours( frame, hull_pt, 0, Scalar(0,0,255), 2, 8, vector<Vec4i>(), 0, Point() );

				//gesture analysis
				if(defects.size()>5)
				{
					vector<Vec4i> index;
					int flag=0;

					for(int i=0;i<defects.size();i++)
					{
						if(defects[i][3]>10000)
						{
							index.push_back(defects[i]);
							if(flag==0)
							{
								circle(frame,contours[largest_contour_index][defects[i][0]],5,Scalar(0,255,0),2);
								flag=1;
							}
							circle(frame,contours[largest_contour_index][defects[i][1]],5,Scalar(0,255,0),2);
							circle(frame,contours[largest_contour_index][defects[i][2]],5,Scalar(0,0,255),2);
						}
					}

					gesture_obj=new gesture(center,index,contours[largest_contour_index]);
					gesture_obj->gesture_main();
					delete(gesture_obj);
				}
			}
		}

		imshow("Mask",mask);
		switch (display)
		{
		case 0:
			imshow("Video",frame);
			break;
		case 1:
			imshow("Video",img);
			break;
		default:
			break;
		}

		char ch=waitKey(1);
		switch (ch)
		{
		case 'x':
			return 1;
		case 'd':
			display=!display;
		default:
			break;
		}

		contours.clear();
		hierarchy.clear();
		contour_area.clear();
	}
	return 0;
}