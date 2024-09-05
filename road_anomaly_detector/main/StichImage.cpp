#include "StichImage.h"



void StichImage::saveAllImages(uint32_t count, char* filename, EImageFileFormat imgeformat)

{
    // does not make scene to use this function with high number of grab results. 
    // this is made just for test purpose only


	if (Imagelist.size() >= count)
    {
        list<Image>::iterator it = Imagelist.begin();
        uint32_t neededMemory = it->m_sizeX * it->m_sizeY *  count;
        uint8_t* m_Buffer = new uint8_t[neededMemory];

        for (int x = 0; x < count; ++x)
        {
            memcpy(m_Buffer + ((neededMemory / count)*x) , it->m_Buffer, neededMemory / count);

            ++it;
        }

        it = Imagelist.begin();
        CPylonImage  temp;
		temp.AttachUserBuffer(m_Buffer, neededMemory, it->m_PixelType, it->m_sizeX, it->m_sizeY *count, it->m_PadingX, it->m_PylonImageOrientation);
		temp.Save(imgeformat, filename);
		delete[] m_Buffer;
		m_Buffer = NULL;
    }

	

}

int StichImage::GetStichedImage(uint32_t NoOfImageToBeStiched, Image & StichedImageIn)
{
    //return 1 if it could successfully deliver needed images.
	// in case the list does not contains enough images to be stitched then the status will be -1.
	
	if (Imagelist.size() >= NoOfImageToBeStiched)
    {
        list <Image> ListTemp;

        mtx_get.lock();
        list<Image>::iterator it = Imagelist.begin();

        for (int x = 0; x < NoOfImageToBeStiched; ++x)
        {
            // make a tem list in order to avoid blocking the main list while memcopy().
            ListTemp.push_back(*it);
            ++it;
            Imagelist.pop_front();  
        }  
        mtx_get.unlock();

        

        if (isAllFramsSameSize)
        { 
            list<Image>::iterator it = ListTemp.begin();

          //  uint8_t* Full image = new uint8_t[NoOfImageToBeStiched * it->m_sizeX * it->m_sizeY];
            size_t Payloadsize = it->m_sizeX * it->m_sizeY;

            int y = 0;
			StichedImageIn.m_PadingX = it->m_PadingX;
			StichedImageIn.m_PixelType = it->m_PixelType;
			StichedImageIn.m_PylonImageOrientation = it->m_PylonImageOrientation;
			StichedImageIn.m_sizeX = it->m_sizeX;
			StichedImageIn.m_sizeY = it->m_sizeY * NoOfImageToBeStiched;
			

            for (it; it != ListTemp.end(); ++it)
            {
				memcpy(StichedImageIn.m_Buffer + (y*Payloadsize), it->m_Buffer, Payloadsize);
                y++;
            }

            it = ListTemp.begin();
            // release the memory from temp list
            int count = ListTemp.size();
            for (int x = 0; x < count; ++x)
            {
                // make sure that the object of buffer is deleted.
                it->releaseBuffer();
                ++it;
            }
            ListTemp.clear();
           
        } 
        else
        {
            //TBD for the case of sequencer in future . sma.
        }

        return 1;

    }
    else
    {
        return -1;
    }
    
}

void StichImage::addGrabResult(IImage& ImageNew)
{
    Image img(ImageNew.GetHeight()*ImageNew.GetWidth());

   // uint8_t* m_Buffer_Test = new uint8_t[ImageNew.GetHeight()*ImageNew.GetWidth()];
   // img.m_Buffer = new uint8_t[];
    img.m_PadingX = ImageNew.GetPaddingX();
    img.m_PixelType = ImageNew.GetPixelType();
    img.m_sizeY = ImageNew.GetHeight();
    img.m_sizeX = ImageNew.GetWidth();
    img.m_PylonImageOrientation = ImageNew.GetOrientation();
    memcpy(img.m_Buffer, ImageNew.GetBuffer(), img.m_sizeY * img.m_sizeX);

    mtx_add.lock();
    Imagelist.push_back(img);

    mtx_add.unlock();
}
