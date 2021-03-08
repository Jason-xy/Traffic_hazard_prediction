#include "jetson-utils/videoSource.h"
#include "jetson-utils/videoOutput.h"
#include "jetson-inference/detectNet.h"
#include <jetson-utils/cudaOverlay.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/imageIO.h>
#include <signal.h>
#include <typeinfo>

using namespace std;


bool signal_recieved = false;

void sig_handler(int signo){
	if(signo == SIGINT){
		LogVerbose("received SIGINT\n");
		signal_recieved = true;
	}
}

int main(int argc, char** argv){
	//parse command line
	commandLine cmdLine(argc, argv, (const char*)NULL);

	//create input stream
	videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));
    if(!input){
		LogError("CarDetection:  failed to create input stream\n");
		return 0;
	}

	//create output stream
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	if(!output)
		LogError("CarDetection:  failed to create output stream\n");

    //create detection network
    detectNet* net = detectNet::Create( "../../dataset/SSD/models/detectnet.prototxt",\
										"../../dataset/SSD/models/onnx_file/open_image_dataset_3_7.onnx",\
										"",\
                                        "../../dataset/SSD/models/onnx_file/labels.txt",\
                                        (float)0.3,\
                                        "input_0",\
										"scores",\
                                        "boxes",\
										DEFAULT_MAX_BATCH_SIZE,\
										TYPE_FASTEST,\
										DEVICE_GPU,\
										(bool)1);
	if(!net){
		LogError("CarDetection:  failed to load detectNet model\n");
		return 0;
	}

	// parse overlay flags
	const uint32_t overlayFlags = detectNet::OverlayFlagsFromStr("box,labels,conf");


	//processing loop
	while( !signal_recieved ){
		// capture next image image
		uchar4* image = NULL;

		if(!input->Capture(&image, 1000)){
			// check for EOS
			if(!input->IsStreaming())
				break; 

			LogError("CarDetection:  failed to capture video frame\n");
			continue;
		}

		// detect objects in the frame
		detectNet::Detection* detections = NULL;

		const int numDetections = net->Detect(image, input->GetWidth(), input->GetHeight(), &detections, overlayFlags);

		if(numDetections > 0){
			LogVerbose("%i objects detected\n", numDetections);
		
			for(int n=0; n < numDetections; n++)
			{
				LogVerbose("detected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
				LogVerbose("bounding box %i  (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(), detections[n].Height()); 
			}
		}

		// render outputs
		if(output != NULL)
		{
			output->Render(imgOutput, input->GetWidth(), input->GetHeight());

			// update the status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, precisionTypeToStr(net->GetPrecision()), net->GetNetworkFPS());
			output->SetStatus(str);

			// check if the user quit
			if(!output->IsStreaming())
				signal_recieved = true;
		}

		// print out timing info
		net->PrintProfilerTimes();
	}
	//destroy resources
	LogVerbose("CarDetection:  shutting down...\n");
	
	SAFE_DELETE(input);
	SAFE_DELETE(output);
	SAFE_DELETE(net);

	LogVerbose("CarDetection:  shutdown complete.\n");
	return 0;
}