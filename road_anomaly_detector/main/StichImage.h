// StichImage.cpp
/*
   this class can be used to stitch multiple grab results into a single image.
   SMA 02.07.2015

*/


#pragma once
#include <thread>         
#include <mutex>  
#include <list>

#include <pylon/PylonIncludes.h>
#include <pylon/PylonGUI.h>

using namespace Pylon;
using namespace std;


class Image 
{
public:
    uint8_t* m_Buffer = NULL;
    PixelType m_PixelType;
    uint32_t m_sizeX = 0;
    uint32_t m_sizeY = 0;
    EImageOrientation m_PylonImageOrientation;
    uint32_t m_PadingX;
    // must be called after object has been removed out of the list.
    void releaseBuffer()
    {
        delete[] m_Buffer;
        m_Buffer = NULL;
    }
  
    
    Image(int64_t Payloadsize)
    {
        m_Buffer = new uint8_t[Payloadsize];
    }
    ~Image()
    {
       // delete [] m_Buffer;
        //m_Buffer = NULL;
    }

};



class StichImage
{
public:
    //normally all images are same size except that sequencer is used to change AOI. In that case false has to be passed
    // currently this class support only 8bit pixel format.
    StichImage(bool) : isAllFramsSameSize( isAllFramsSameSize){};

    void addGrabResult(IImage& newImagetoBeKept);
    
    // fills the StichedImageIn with number of the wished grabresult. returns -1 if the list does not contains 
    // enough grabresults.
	int GetStichedImage(uint32_t NoOfImageToBeStiched, Image & StichedImageIn);

    ~StichImage(void)
    {
        if (Imagelist.size() > 0)
        {
            int count = Imagelist.size();
            list<Image>::iterator it = Imagelist.begin();
            for (int x = 0; x < count; ++x)
            {
                // make sure that the object of buffer is deleted.
                it->releaseBuffer();
                ++it;
            }
            Imagelist.clear();
        }
      
    };

	void saveAllImages(uint32_t count, char* filename, EImageFileFormat imgeformat);


private:
    bool isAllFramsSameSize = true;
    mutex mtx_add,mtx_get;

    // keep all images that has bees add via the function addGrabResult();
    // Default will simply delete all images that have not been released. 
    list <Image> Imagelist;

};

