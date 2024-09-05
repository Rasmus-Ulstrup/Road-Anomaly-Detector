// Grab.cpp
/*
    Note: Before getting started, Basler recommends reading the Programmer's Guide topic
    in the pylon C++ API documentation that gets installed with pylon.
    If you are upgrading to a higher major version of pylon, Basler also
    strongly recommends reading the Migration topic in the pylon C++ API documentation.

    This sample illustrates how to grab and process images using the CInstantCamera class.
    The images are grabbed and processed asynchronously, i.e.,
    while the application is processing a buffer, the acquisition of the next buffer is done
    in parallel.

    The CInstantCamera class uses a pool of buffers to retrieve image data
    from the camera device. Once a buffer is filled and ready,
    the buffer can be retrieved from the camera object for processing. The buffer
    and additional image data are collected in a grab result. The grab result is
    held by a smart pointer after retrieval. The buffer is automatically reused
    when explicitly released or when the smart pointer object is destroyed.
*/

// Include files to use the PYLON API.
#include <pylon/PylonIncludes.h>
#ifdef PYLON_WIN_BUILD
#    include <pylon/PylonGUI.h>
#endif

#include <thread>

#include "StichImage.h"

// Namespace for using pylon objects.
using namespace Pylon;

// Namespace for using cout.
using namespace std;
using namespace GenApi;

// Number of images to be grabbed.
static const uint32_t c_countOfImagesToGrab = 120;
bool bDone = false;

void DisplayStichedImages(int NoOfImagestoBeStiched,int64_t PayloadSize, StichImage* _StichImage)
{
	//use 2 image to keep bit long in the display
	Image IMG(PayloadSize * NoOfImagestoBeStiched);
	int status = false;
	
	
	while (!bDone)
	{
		
	    status = _StichImage->GetStichedImage(3, IMG);

		if (status > 0)
		{
			CPylonImage pylonimage;
			pylonimage.AttachUserBuffer(IMG.m_Buffer, PayloadSize*NoOfImagestoBeStiched, IMG.m_PixelType, IMG.m_sizeX, IMG.m_sizeY, IMG.m_PadingX, IMG.m_PylonImageOrientation);
			Pylon::DisplayImage(2, pylonimage);
		}
		else
		{
			Sleep(5000);
		}
			
	
	}

	IMG.releaseBuffer();

}


int main(int argc, char* argv[])
{
    // The exit code of the sample application.
    int exitCode = 0;

	// the parameter true tells that all images are at the same size.
	// this is reserved to extend the class also for sequencer mode.
	// set the frame hight on the camera so that target frame size dividable 
	// e.g You wish a target frame size 9000 then frame height of camera 1000 or 3000. In this case call 
	// Images->GetStichedImage passing the parameter 9 or 3.
	StichImage* Images = new  StichImage(true);
	

    // Automagically call PylonInitialize and PylonTerminate to ensure the pylon runtime system
    // is initialized during the lifetime of this object.
    Pylon::PylonAutoInitTerm autoInitTerm;

    try
    {
        // Create an instant camera object with the camera device found first.
        CInstantCamera camera( CTlFactory::GetInstance().CreateFirstDevice());
		camera.Open();

		INodeMap& nodemap = camera.GetNodeMap();

		CIntegerPtr Height(nodemap.GetNode("Height"));
		CIntegerPtr	Width(nodemap.GetNode("Width"));
		CIntegerPtr	PayloadSize(nodemap.GetNode("PayloadSize"));

		Width->SetValue(Width->GetMax());
		if (Height->GetMax() > 3000)
		{
			Height->SetValue(3000);
		}
		
		int64_t Payloadsize = PayloadSize->GetValue();
		

		thread ThDisplyImage(DisplayStichedImages, 3, Payloadsize, Images);

        // Print the model name of the camera.
        cout << "Using device " << camera.GetDeviceInfo().GetModelName() << endl;

        // The parameter MaxNumBuffer can be used to control the count of buffers
        // allocated for grabbing. The default value of this parameter is 10.
        camera.MaxNumBuffer = 20;

        // Start the grabbing of c_countOfImagesToGrab images.
        // The camera device is parameterized with a default configuration which
        // sets up free-running continuous acquisition.
        camera.StartGrabbing( c_countOfImagesToGrab);

        // This smart pointer will receive the grab result data.
        CGrabResultPtr ptrGrabResult;

        // Camera.StopGrabbing() is called automatically by the RetrieveResult() method
        // when c_countOfImagesToGrab images have been retrieved.
        while ( camera.IsGrabbing())
        {
            // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            camera.RetrieveResult( 5000, ptrGrabResult, TimeoutHandling_ThrowException);

            // Image grabbed successfully?
            if (ptrGrabResult->GrabSucceeded())
            {
                // Access the image data.
                cout << "SizeX: " << ptrGrabResult->GetWidth() << endl;
                cout << "SizeY: " << ptrGrabResult->GetHeight() << endl;
                const uint8_t *pImageBuffer = (uint8_t *) ptrGrabResult->GetBuffer();
                cout << "Gray value of first pixel: " << (uint32_t) pImageBuffer[0] << endl << endl;

				// now you feed the camera result in to the list.
				Images->addGrabResult(ptrGrabResult);

#ifdef PYLON_WIN_BUILD
                // Display the grabbed image.
                Pylon::DisplayImage(1, ptrGrabResult);
#endif
            }
            else
            {
                cout << "Error: " << ptrGrabResult->GetErrorCode() << " " << ptrGrabResult->GetErrorDescription() << endl;
            }
        }

		bDone = true;

		ThDisplyImage.join();
    }
    catch (GenICam::GenericException &e)
    {
		bDone = true;
		// Error handling.
        cerr << "An exception occurred." << endl
        << e.GetDescription() << endl;
        exitCode = 1;
    }


    // Comment the following two lines to disable waiting on exit.
    cerr << endl << "Press Enter to exit." << endl;
    while( cin.get() != '\n');

    return exitCode;
}
