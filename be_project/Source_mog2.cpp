#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <time.h>

#include <opencv2\core\core.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\ocl\ocl.hpp>
#include <opencv2\ml\ml.hpp>

#include <Windows.h>

using namespace std;
using namespace cv;
using namespace cv::ocl;

class gesture
{
	vector<Vec4i> defects;
	vector<Point> contour;
	Point2f center;
public:
	int fingers;
	gesture(Point2f cent,vector<Vec4i> index,vector<Point> cont)
	{
		center=cent;
		defects=index;
		contour=cont;
		fingers=0;
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
		x-=mx;
		y+=my;
		SetCursorPos(x,y);
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
	namedWindow("Mask",CV_WINDOW_AUTOSIZE);

	Mat frame,mask,palm_mask,img;
	oclMat oclframe,oclmask,oclimg;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<double> contour_area;
	vector<vector<Point> > palm_contours;
	vector<Vec4i> palm_hierarchy;
	vector<double> palm_contour_area;
	vector<uchar> outbuf;
	Moments m;
	Point2f center;
	Point2f center_old=Point2f(0,0);
	int display=0;
	int largest_contour_index;
	int palm_largest_contour_index;
	int prev_value=0;
	int prev_times=0;
	int update_bg_model=-1;
	bool gesture_toggle=false;
	bool mouse_toggle=false;
	bool paint_toggle=false;

	gesture *gesture_obj;
	mouse mouse_obj;
	ocl::MOG2 mog2;
	img=Mat(480,640,CV_8UC3);

	// fps counter begin
    time_t start, end;
    int counter = 0;
    double sec;
    double fps;
    // fps counter end

	while(true)
	{
		// fps counter begin
        if (counter == 0){
            time(&start);
        }
        // fps counter end

		cap>>frame;
		if(frame.empty())
			continue;

		//gpu init
		oclframe.upload(frame);
		mog2(oclframe,oclmask,update_bg_model);
		mog2.getBackgroundImage(oclimg);

		//create mask
		ocl::erode(oclmask,oclmask,getStructuringElement(MORPH_ELLIPSE,Size(5,5),Point(2,2)));
		ocl::dilate(oclmask,oclmask,getStructuringElement(MORPH_ELLIPSE,Size(5,5),Point(2,2)));
		oclmask.download(mask);
		ocl::erode(oclmask,oclmask,getStructuringElement(MORPH_ELLIPSE,Size(9,9),Point(4,4)));
		ocl::erode(oclmask,oclmask,getStructuringElement(MORPH_ELLIPSE,Size(9,9),Point(4,4)));
		ocl::erode(oclmask,oclmask,getStructuringElement(MORPH_ELLIPSE,Size(9,9),Point(4,4)));
		ocl::erode(oclmask,oclmask,getStructuringElement(MORPH_ELLIPSE,Size(9,9),Point(4,4)));
		ocl::dilate(oclmask,oclmask,getStructuringElement(MORPH_ELLIPSE,Size(9,9),Point(4,4)));
		oclmask.download(palm_mask);

		//find contours
		Mat maskcpy=mask.clone();
		findContours(maskcpy,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
		Mat palm_maskcpy=palm_mask.clone();
		findContours(palm_maskcpy,palm_contours,palm_hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
		vector<vector<int> >hull(1);
		vector<vector<Point> >hull_pt(1);
		vector<Vec4i> defects;

		//get mouse position
		mouse_obj.getmousepos();

		if(!contours.empty() && !palm_contours.empty())
		{
			//find largest contour
			for(int i=0;i<contours.size();i++)
			{
				contour_area.push_back(contourArea(contours[i]));
			}
			largest_contour_index=distance(contour_area.begin(),max_element(contour_area.begin(),contour_area.end()));

			for(int i=0;i<palm_contours.size();i++)
			{
				palm_contour_area.push_back(contourArea(palm_contours[i]));
			}
			palm_largest_contour_index=distance(palm_contour_area.begin(),max_element(palm_contour_area.begin(),palm_contour_area.end()));

			if(contour_area[largest_contour_index]>1000)
			{
				drawContours(frame,contours,largest_contour_index,Scalar(255,0,0),2);

				Rect palm_myrect=boundingRect(palm_contours[palm_largest_contour_index]);
				if(palm_myrect.width*1.2<palm_myrect.height)
					palm_myrect.height=palm_myrect.width*1.2;
				//rectangle(frame,myrect,Scalar(0,0,0),2);
				//rectangle(frame,palm_myrect,Scalar(0,0,0),2);

				vector<Point> mycontour;
				for(int i=0;i<contours[largest_contour_index].size();i++)
				{
					if(contours[largest_contour_index][i].y<(palm_myrect.y+palm_myrect.height))
						mycontour.push_back(contours[largest_contour_index][i]);
				}
				contours[largest_contour_index]=mycontour;

				if(!mycontour.empty())
				{
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

					if(gesture_toggle)
					{
						char num[10];
						_itoa(gesture_obj->fingers,num,10);
						putText(frame,num,Point(100,50),CV_FONT_HERSHEY_SCRIPT_SIMPLEX,1,Scalar(100,100,100),2);
					}

					if(mouse_toggle)
					{
						switch (gesture_obj->fingers)
						{
						case 0:
							mouse_obj.updatemousepos(center.x-center_old.x,center.y-center_old.y);
							break;
						case 1:
							if(mouse_obj.flag==0)
							{
								if(prev_value==1)
									prev_times++;
								else 
								{
									prev_value=1;
									prev_times=0;
								}
								if(prev_times==10)
								{
									mouse_obj.MouseLeftClick();
									prev_times=0;
								}
							}
							break;
						case 2:
							if(mouse_obj.flag==0)
							{
								if(prev_value==2)
									prev_times++;
								else 
								{
									prev_value=2;
									prev_times=0;
								}
								if(prev_times==10)
								{
									mouse_obj.MouseDoubleClick();
									prev_times=0;
								}
							}
							break;
						case 3:
							if(mouse_obj.flag==0)
							{
								if(prev_value==3)
									prev_times++;
								else 
								{
									prev_value=3;
									prev_times=0;
								}
								if(prev_times==10)
								{
									mouse_obj.MouseRightClick();
									prev_times=0;
								}
							}
							break;
						case 4:
							if(mouse_obj.flag==0)
							{
								if(prev_value==4)
									prev_times++;
								else 
								{
									prev_value=4;
									prev_times=0;
								}
								if(prev_times==10)
								{
									mouse_obj.mouse_hold_toggle();
									prev_times=0;
								}
							}
							break;
						case 5:
							mouse_obj.flag=0;
							break;
						default:
							break;
						}
					}

					delete(gesture_obj);
				}

				if(paint_toggle)
				{
					line(img,center_old,center,Scalar(0,0,255),2);
				}

				center_old=center;
				}
			}
		}

		// fps counter begin
        time(&end);
        counter++;
        sec = difftime(end, start);
        fps = counter/sec;
        if (counter > 30)
        {
			char num[10];
			_itoa(fps,num,10);
			putText(frame,num,Point(10,50),CV_FONT_HERSHEY_SCRIPT_SIMPLEX,1,Scalar(0,0,100),2);
		}

		// overflow protection
        if (counter == (INT_MAX - 1000))
            counter = 0;
        // fps counter end

		imshow("Video",frame);
		switch (display)
		{
		case 0:
			imshow("Mask",mask);
			break;
		case 1:
			imshow("Mask",palm_mask);
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
		case 'g':
			gesture_toggle=!gesture_toggle;
			break;
		case 'm':
			mouse_toggle=!mouse_toggle;
			break;
		case 'p':
			paint_toggle=!paint_toggle;
			break;
		case 'c':
			img=Mat(480,640,CV_8UC3);
			break;
		case 'r':
			update_bg_model=-1;
			break;
		case 's':
			update_bg_model=0;
			break;
		default:
			break;
		}

		contours.clear();
		hierarchy.clear();
		contour_area.clear();
		palm_contours.clear();
		palm_hierarchy.clear();
		palm_contour_area.clear();
		outbuf.clear();
	}
	return 0;
}