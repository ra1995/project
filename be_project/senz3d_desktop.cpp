#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <omp.h>

#include <Windows.h>

#include <pxcsession.h>
#include <pxcsmartptr.h>
#include <pxccapture.h>
#include <pxcprojection.h>
#include <pxcmetadata.h>
#include <util_pipeline.h>

#include <opencv2\core\core.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\ocl\ocl.hpp>
#include <opencv2\ml\ml.hpp>

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
		/*double mean_defect_distance=0;

		for(int i=0;i<defects.size();i++)
		{
			mean_defect_distance+=distance(contour[defects[i][2]],center);
		}
		mean_defect_distance/=defects.size();*/
		
		for(int i=0;i<defects.size();i++)
		{
			if(i==0)
				if(distance(contour[defects[0][0]],center)>(2*distance(contour[defects[i][2]],center)))
				{
					fingers++;
				}
			if(distance(contour[defects[i][1]],center)>(2*distance(contour[defects[i][2]],center)))
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
	void updatemousepos(int mx,int my,float s)
	{
		getmousepos();
		x-=s*mx;
		y+=s*my;
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
	UtilPipeline pp;
	pp.EnableImage(PXCImage::COLOR_FORMAT_RGB24,640,480);
	pp.EnableImage(PXCImage::COLOR_FORMAT_DEPTH,320,240);
	pp.Init();

	namedWindow("Video",CV_WINDOW_AUTOSIZE);
	namedWindow("Mask",CV_WINDOW_AUTOSIZE);

	Mat frame,mask,palm_mask;
	Mat depth_map,depth_thresh,depth_rgb;
	oclMat oclmask,oclframe,oclscale;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<double> contour_area;
	vector<vector<Point> > palm_contours;
	vector<Vec4i> palm_hierarchy;
	vector<double> palm_contour_area;
	Moments m;
	Point2f center;
	Point2f center_old=Point2f(0,0);
	int display=0;
	int largest_contour_index;
	int palm_largest_contour_index;
	int prev_value=0;
	int prev_times=0;
	double minval,maxval;
	bool gesture_toggle=false;
	bool mouse_toggle=false;
	bool paint_toggle=false;

	gesture *gesture_obj;
	mouse mouse_obj;
	float sense[3]={1.0,1.5,2.0};
	int sensitivity=2;

	UtilCapture *capture;
	PXCSession *session;
	PXCCapture::Device *device;

	session=pp.QuerySession();
	capture=pp.QueryCapture();
	device=capture->QueryDevice();
	pxcUID uid;
	device->QueryPropertyAsUID(PXCCapture::Device::PROPERTY_PROJECTION_SERIALIZABLE, &uid);

	while(true)
	{
		if (!pp.AcquireFrame(true))
			break;
		PXCImage *color=pp.QueryImage(PXCImage::IMAGE_TYPE_COLOR);
		PXCImage *depth=pp.QueryImage(PXCImage::IMAGE_TYPE_DEPTH);
		PXCImage::ImageData data_rgb;
		PXCImage::ImageData data_depth;

		PXCProjection *pj;
		session->DynamicCast<PXCMetadata>()->CreateSerializable<PXCProjection>(uid,&pj);

		color->AcquireAccess(PXCImage::ACCESS_READ,&data_rgb);
		depth->AcquireAccess(PXCImage::ACCESS_READ,&data_depth);

		frame=Mat(480,640,CV_8UC3,(uchar*)data_rgb.planes[0],640*3*sizeof(uchar)).clone();
		depth_map=Mat(240,320,CV_16U,(uchar*)data_depth.planes[0]).clone();
		resize(depth_map,depth_map,Size(640,480));
		//resize(frame,frame,Size(960,720));
		inRange(depth_map,Scalar(350),Scalar(700),depth_thresh);

		color->ReleaseAccess(&data_rgb);
		depth->ReleaseAccess(&data_depth);
		pp.ReleaseFrame(); 

		//gpu init
		oclmask.upload(depth_thresh);
		oclframe.upload(depth_map);

		//create mask
		ocl::erode(oclmask,oclmask,getStructuringElement(MORPH_ELLIPSE,Size(5,5),Point(2,2)));
		ocl::minMax(oclframe,&minval,&maxval,oclmask);
		inRange(depth_map,Scalar(minval-10),Scalar(minval+100),depth_thresh);
		//inRange(depth_map,Scalar(maxval-100),Scalar(maxval+10),depth_thresh);
		depth_thresh.convertTo(depth_thresh,CV_8UC1);
		cvtColor(depth_thresh,depth_rgb,CV_GRAY2BGR);
		oclmask.upload(depth_thresh);
		ocl::erode(oclmask,oclmask,getStructuringElement(MORPH_ELLIPSE,Size(5,5),Point(2,2)));
		oclmask.download(mask);

		ocl::erode(oclmask,oclmask,getStructuringElement(MORPH_ELLIPSE,Size(9,9),Point(4,4)));
		ocl::erode(oclmask,oclmask,getStructuringElement(MORPH_ELLIPSE,Size(9,9),Point(4,4)));
		ocl::erode(oclmask,oclmask,getStructuringElement(MORPH_ELLIPSE,Size(9,9),Point(4,4)));
		ocl::dilate(oclmask,oclmask,getStructuringElement(MORPH_ELLIPSE,Size(9,9),Point(4,4)));
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

			if(contour_area[largest_contour_index]>1000 && contours[largest_contour_index].size()>3)
			{
				drawContours(depth_rgb,contours,largest_contour_index,Scalar(255,0,0),2);

				RotatedRect rotated_palm_myrect=minAreaRect(palm_contours[palm_largest_contour_index]);
				RotatedRect rotated_myrect=minAreaRect(contours[largest_contour_index]);
				Point2f rect_points[4],myrect_points[4];
				rotated_palm_myrect.points(rect_points);
				rotated_myrect.points(myrect_points);

				float rht,rwd,angle;
				float dist;
				float rx,ry;
				rht=max(rotated_palm_myrect.size.height,rotated_palm_myrect.size.width);
				rwd=min(rotated_palm_myrect.size.height,rotated_palm_myrect.size.width);
				dist=(rht/2)-(rwd/2);
				rx=rotated_palm_myrect.center.x;
				ry=rotated_palm_myrect.center.y;
				angle=-1*rotated_palm_myrect.angle;
				if(rotated_palm_myrect.size.height>rotated_palm_myrect.size.width)
				{
					angle=90-angle;
				}

				if(rotated_palm_myrect.size.height>rotated_palm_myrect.size.width)
				{
					rx=rx-(dist*cos(angle*(3.14/180)));
					ry=ry-(dist*sin(angle*(3.14/180)));
				}
				else{
					rx=rx+(dist*cos(angle*(3.14/180)));
					ry=ry-(dist*sin(angle*(3.14/180)));
				}
				circle(depth_rgb,Point2f(rx,ry),2,Scalar(0,255,0),2);

				rotated_palm_myrect.center=Point2f(rx,ry);
				if(rwd*1.5<rht)
				{
					rotated_palm_myrect.size=Size2f(rwd,rwd);
					Point2f rect_points_new[4];
					vector<Point2f> pts(3);
					rotated_palm_myrect.points(rect_points_new);

					for(int i=0;i<4;i++)
					{
						line(depth_rgb,myrect_points[i],myrect_points[(i+1)%4],Scalar(255,0,0),2);
						line(depth_rgb,rect_points[i],rect_points[(i+1)%4],Scalar(0,255,0),2);
						line(depth_rgb,rect_points_new[i],rect_points_new[(i+1)%4],Scalar(0,0,255),2);
					}

					float m,c;
					if(gesture_obj->distance(rect_points[1],rect_points_new[1])<3)
					{
						rotated_palm_myrect.size=Size2f(rwd*1.5,rwd*1.5);
						rotated_palm_myrect.points(rect_points_new);
						m=(rect_points_new[0].y-rect_points_new[3].y)/(rect_points_new[0].x-rect_points_new[3].x);
						c=rect_points_new[0].y-(m*rect_points_new[0].x);
						line(depth_rgb,rect_points_new[0],rect_points_new[3],Scalar(255,0,0),2);
					}
					else
					{
						rotated_palm_myrect.size=Size2f(rwd*1.5,rwd*1.5);
						rotated_palm_myrect.points(rect_points_new);
						m=(rect_points_new[0].y-rect_points_new[1].y)/(rect_points_new[0].x-rect_points_new[1].x);
						c=rect_points_new[0].y-(m*rect_points_new[0].x);
						line(depth_rgb,rect_points_new[0],rect_points_new[1],Scalar(255,0,0),2);
					}

					vector<Point> mycontour;
					for(int i=0;i<contours[largest_contour_index].size();i++)
					{
						if(((m*contours[largest_contour_index][i].x)-contours[largest_contour_index][i].y+c)>=0)
							mycontour.push_back(contours[largest_contour_index][i]);
					}

					contours[largest_contour_index]=mycontour;
				}

				if(contours[largest_contour_index].size()>3)
				{
				//find center of contour
				m=moments(contours[largest_contour_index]);
				center.x=m.m10/m.m00;
				center.y=m.m01/m.m00;
				circle(depth_rgb,center,5,Scalar(0,255,0),2);

				//map 3d to 2d
				PXCPoint3DF32 dcenter = { (center.x/640)*320, (center.y/480)*240, depth_map.at<unsigned short int>(center)};
				PXCPointF32 ccenter;
				pj->MapDepthToColorCoordinates(1,&dcenter,&ccenter);
				circle(frame,Point2f(ccenter.x,ccenter.y),5,Scalar(0,255,0),2);

				//find convex hull
				convexHull( Mat(contours[largest_contour_index]), hull[0], false );
				convexHull( Mat(contours[largest_contour_index]), hull_pt[0], false );
				convexityDefects(contours[largest_contour_index], hull[0], defects);
				drawContours( depth_rgb, hull_pt, 0, Scalar(0,0,255), 2, 8, vector<Vec4i>(), 0, Point() );

				//gesture analysis
				if(defects.size()>5)
				{
					vector<Vec4i> index;
					int flag=0;

					for(int i=0;i<defects.size();i++)
					{
						if(defects[i][3]>(15000-(15*depth_map.at<unsigned short int>(center))))
						{
							index.push_back(defects[i]);
							if(flag==0)
							{
								circle(depth_rgb,contours[largest_contour_index][defects[i][0]],5,Scalar(0,255,0),2);
								flag=1;
							}
							circle(depth_rgb,contours[largest_contour_index][defects[i][1]],5,Scalar(0,255,0),2);
							circle(depth_rgb,contours[largest_contour_index][defects[i][2]],5,Scalar(0,0,255),2);
						}
					}

					gesture_obj=new gesture(Point2f(rx,ry),index,contours[largest_contour_index]);
					gesture_obj->gesture_main();

					if(gesture_obj->fingers==0)
					{
						RotatedRect hand_myrect=minAreaRect(contours[largest_contour_index]);
						if(max(hand_myrect.size.height,hand_myrect.size.width)>(1.7*min(hand_myrect.size.height,hand_myrect.size.width)))
							gesture_obj->fingers=1;
					}

					if(gesture_toggle)
					{
						char num[10];
						_itoa(gesture_obj->fingers,num,10);
						putText(frame,num,Point(100,50),CV_FONT_HERSHEY_SCRIPT_SIMPLEX,1,Scalar(0,0,255),2);
					}

					if(mouse_toggle)
					{
						switch (gesture_obj->fingers)
						{
						case 0:
							mouse_obj.updatemousepos(center.x-center_old.x,center.y-center_old.y,sense[sensitivity]);
							mouse_obj.flag=0;
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

				center_old=center;
				}
			}
		}

		pj->Release();
		imshow("Video",frame);
		switch (display)
		{
		case 0:
			imshow("Mask",depth_rgb);
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
			pp.Close();
			return 1;
		case 'd':
			display=!display;
			break;
		case 'g':
			gesture_toggle=!gesture_toggle;
			break;
		case 'm':
			mouse_toggle=!mouse_toggle;
			break;
		case 'p':
			paint_toggle=!paint_toggle;
			break;
		case 's':
			sensitivity=(sensitivity+1)%3;
			cout<<"Sensitivity :"<<sense[sensitivity]<<"\n";
		default:
			break;
		}

		contours.clear();
		hierarchy.clear();
		contour_area.clear();
		palm_contours.clear();
		palm_hierarchy.clear();
		palm_contour_area.clear();
		frame.release();
	}
	pp.Close();
	return 0;
}