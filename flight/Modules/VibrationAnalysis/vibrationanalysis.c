/**
 ******************************************************************************
 * @addtogroup TauLabsModules Tau Labs Modules
 * @{
 * @addtogroup VibrationAnalysisModule Vibration analysis module
 * @{
 *
 * @file       vibrationanalysis.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013
 * @brief      Performs an FFT on the accels to estimation vibration
 *
 * @see        The GNU Public License (GPL) Version 3
 *
 *****************************************************************************/
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

/**
 * Input objects: @ref Accels, @ref VibrationAnalysisSettings
 * Output object: @ref VibrationAnalysisOutput
 *
 * This module executes on a timer trigger. When the module is
 * triggered it will update the data of VibrationAnalysiOutput, based on
 * the output of an FFT running on the accelerometer samples. 
 */

#include "openpilot.h"
#include "physical_constants.h"
#include "arm_math.h"

#include "accels.h"
#include "modulesettings.h"
#include "vibrationanalysisoutput.h"
#include "vibrationanalysissettings.h"


// Private constants

#define MAX_QUEUE_SIZE 2
#define STACK_SIZE_BYTES (200 + 484 + (13*fft_window_size)*0) // The fft memory requirement grows linearly 
																				  // with window size. The constant is multiplied
																				  // by 0 in order to reflect the fact that the
																				  // malloc'ed memory is not taken from the module 
																				  // but instead from the heap. Nonetheless, we 
																				  // can know a priori how much RAM this module 
																				  // will take.
#define TASK_PRIORITY (tskIDLE_PRIORITY+1)
#define SETTINGS_THROTTLING_MS 100

#define MAX_ACCEL_RANGE 16                          // Maximum accelerometer resolution in [g]
#define FLOAT_TO_Q15 (32768/(MAX_ACCEL_RANGE*GRAVITY)) // This is the scaling constant that scales all input floats to +-

// Private variables
static xTaskHandle taskHandle;
static xQueueHandle queue;
static bool module_enabled = false;

static struct VibrationAnalysis_data {
	uint16_t accels_sum_count;
	uint16_t fft_window_size;
	uint8_t num_upscale_bits;

	float accels_data_sum_x;
	float accels_data_sum_y;
	float accels_data_sum_z;
	
	float accels_static_bias_x; // In all likelyhood, the initial values will be close to 
	float accels_static_bias_y; // (0,0,-g). In the case where they are not, this will still  
	float accels_static_bias_z; // converge to the true bias in a few thousand measurements.	
	
	int16_t *accel_buffer_complex_x_q15;
	int16_t *accel_buffer_complex_y_q15;
	int16_t *accel_buffer_complex_z_q15;
	
	int16_t *fft_output;
} *vtd;


// Private functions
static void VibrationAnalysisTask(void *parameters);

/**
 * Start the module, called on startup
 */
static int32_t VibrationAnalysisStart(void)
{
	
	if (!module_enabled)
		return -1;

	//Get the FFT window size
	uint16_t fft_window_size; // Make a local copy in order to check settings before allocating memory
	uint8_t num_upscale_bits;
	VibrationAnalysisSettingsFFTWindowSizeOptions fft_window_size_enum;
	VibrationAnalysisSettingsFFTWindowSizeGet(&fft_window_size_enum);
	switch (fft_window_size_enum) {
		case VIBRATIONANALYSISSETTINGS_FFTWINDOWSIZE_16:
			fft_window_size = 16;
			num_upscale_bits = 4;
			break;
		case VIBRATIONANALYSISSETTINGS_FFTWINDOWSIZE_64:
			fft_window_size = 64;
			num_upscale_bits = 6;
			break;
		case VIBRATIONANALYSISSETTINGS_FFTWINDOWSIZE_256:
			fft_window_size = 256;
			num_upscale_bits = 8;
			break;
		case VIBRATIONANALYSISSETTINGS_FFTWINDOWSIZE_1024:
			fft_window_size = 1024;
			num_upscale_bits = 10;
			break;
		default:
			//This represents a serious configuration error. Do not start module.
			module_enabled = false;
			return -1;
			break;
	}
	

	// Create instances for vibration analysis. Start from i=1 because the first instance is generated
	// by VibrationAnalysisOutputInitialize(). Generate three times the length because there are three
	// vectors. Generate half the length because the FFT output is symmetric about the mid-frequency, 
	// so there's no point in using memory additional memory.
	for (int i=1; i < (fft_window_size>>1); i++) {
		uint16_t ret = VibrationAnalysisOutputCreateInstance();
		if (ret == 0) {
			// This fails when it's a metaobject. Not a very helpful test.
			module_enabled = false;
			return -1;
		}
	}
	
	if (VibrationAnalysisOutputGetNumInstances() != (fft_window_size>>1)){
		// This is a more useful test for failure.
		module_enabled = false;
		return -1;
	}
	
	
	// Allocate and initialize the static data storage only if module is enabled
	vtd = (struct VibrationAnalysis_data *) pvPortMalloc(sizeof(struct VibrationAnalysis_data));
	if (vtd == NULL) {
		module_enabled = false;
		return -1;
	}
	
	// make sure that all struct values are zeroed...
	memset(vtd, 0, sizeof(struct VibrationAnalysis_data));
	//... except for Z axis static bias
	vtd->accels_static_bias_z=-GRAVITY; // [See note in definition of VibrationAnalysis_data structure]

	// Now place the fft window size and number of upscale bits variables into the buffer
	vtd->fft_window_size = fft_window_size;
	vtd->num_upscale_bits = num_upscale_bits;
	
	// Allocate ouput vector
	vtd->fft_output = (int16_t *) pvPortMalloc(fft_window_size*2*sizeof(typeof(*(vtd->fft_output))));
	if (vtd->fft_output == NULL) {
		module_enabled = false; //Check if allocation succeeded
		return -1;
	}
	
	//Create the buffers. They are in Q15 format.
	vtd->accel_buffer_complex_x_q15 = (int16_t *) pvPortMalloc(fft_window_size*2*sizeof(typeof(*vtd->accel_buffer_complex_x_q15)));
	if (vtd->accel_buffer_complex_x_q15 == NULL) {
		module_enabled = false; //Check if allocation succeeded
		return -1;
	}
	vtd->accel_buffer_complex_y_q15 = (int16_t *) pvPortMalloc(fft_window_size*2*sizeof(typeof(*vtd->accel_buffer_complex_y_q15)));
	if (vtd->accel_buffer_complex_y_q15 == NULL) {
		module_enabled = false; //Check if allocation succeeded
		return -1;
	}
	vtd->accel_buffer_complex_z_q15 = (int16_t *) pvPortMalloc(fft_window_size*2*sizeof(typeof(*vtd->accel_buffer_complex_z_q15)));
	if (vtd->accel_buffer_complex_z_q15 == NULL) {
		module_enabled = false; //Check if allocation succeeded
		return -1;
	}
	
	// Start main task
	xTaskCreate(VibrationAnalysisTask, (signed char *)"VibrationAnalysis", STACK_SIZE_BYTES/4, NULL, TASK_PRIORITY, &taskHandle);
	TaskMonitorAdd(TASKINFO_RUNNING_VIBRATIONANALYSIS, taskHandle);
	return 0;
}


/**
 * Initialise the module, called on startup
 */
static int32_t VibrationAnalysisInitialize(void)
{
	ModuleSettingsInitialize();
	
#ifdef MODULE_VibrationAnalysis_BUILTIN
	module_enabled = true;
#else
	uint8_t module_state[MODULESETTINGS_ADMINSTATE_NUMELEM];
	ModuleSettingsAdminStateGet(module_state);
	if (module_state[MODULESETTINGS_ADMINSTATE_VIBRATIONANALYSIS] == MODULESETTINGS_ADMINSTATE_ENABLED) {
		module_enabled = true;
	} else {
		module_enabled = false;
	}
#endif
	
	if (!module_enabled) //If module not enabled...
		return -1;

	// Initialize UAVOs
	VibrationAnalysisSettingsInitialize();
	VibrationAnalysisOutputInitialize();
		
	// Create object queue
	queue = xQueueCreate(MAX_QUEUE_SIZE, sizeof(UAVObjEvent));
		
	return 0;
	
}
MODULE_INITCALL(VibrationAnalysisInitialize, VibrationAnalysisStart)


static void VibrationAnalysisTask(void *parameters)
{
#define MAX_BLOCKSIZE   2048
	
	portTickType lastSysTime;
	portTickType lastSettingsUpdateTime;
	uint8_t runAnalysisFlag = VIBRATIONANALYSISSETTINGS_TESTINGSTATUS_OFF; // By default, turn analysis off
	uint16_t sampleRate_ms = 100; // Default sample rate of 100ms
	uint8_t sample_count;
	UAVObjEvent ev;
	
	// Listen for updates.
	AccelsConnectQueue(queue);
	
	// Declare FFT structure and status variable
	arm_cfft_radix4_instance_q15 cfft_instance;
	arm_status status;
	
	// Initialize the CFFT/CIFFT module
	status = ARM_MATH_SUCCESS;
	bool ifftFlag = false;
	bool doBitReverse = 1;
	status = arm_cfft_radix4_init_q15(&cfft_instance, vtd->fft_window_size, ifftFlag, doBitReverse);
	
	
/** These values are useful for insight into the Fourier transform performed by this module.
	float freq_sample = 1.0f/sampleRate_ms;
	float freq_nyquist = f_s/2.0f;
	uint16_t num_samples = vtd->fft_window_size;
 */

	// Main task loop
	VibrationAnalysisOutputData vibrationAnalysisOutputData;
	sample_count = 0;
	lastSysTime = xTaskGetTickCount();
	lastSettingsUpdateTime = xTaskGetTickCount() - MS2TICKS(SETTINGS_THROTTLING_MS);

	
	// Main module task, never exit from while loop
	while(1)
	{
		// Only check settings once every 100ms
		if(xTaskGetTickCount() - lastSettingsUpdateTime > MS2TICKS(SETTINGS_THROTTLING_MS)){
			//First check if the analysis is active
			VibrationAnalysisSettingsTestingStatusGet(&runAnalysisFlag);
			
			// Get sample rate
			VibrationAnalysisSettingsSampleRateGet(&sampleRate_ms);
			sampleRate_ms = sampleRate_ms > 0 ? sampleRate_ms : 1; //Ensure sampleRate never is 0.
			
			lastSettingsUpdateTime = xTaskGetTickCount();
		}
		
		// If analysis is turned off, delay and then loop.
		if (runAnalysisFlag == VIBRATIONANALYSISSETTINGS_TESTINGSTATUS_OFF) {
			vTaskDelay(200);
			continue;
		}
		
		// Wait until the Accels object is updated, and never time out
		if ( xQueueReceive(queue, &ev, portMAX_DELAY) == pdTRUE )
		{
			/**
			 * Accumulate accelerometer data. This would be a great place to add a 
			 * high-pass filter, in order to eliminate the DC bias from gravity.
			 * Until then, a DC bias subtraction has been added in the main loop.
			 */
			
			AccelsData accels_data;
			AccelsGet(&accels_data);
			
			vtd->accels_data_sum_x += accels_data.x;
			vtd->accels_data_sum_y += accels_data.y;
			vtd->accels_data_sum_z += accels_data.z;
			
			vtd->accels_sum_count++;
		}
		
		// If not enough time has passed, keep accumulating data
		if(xTaskGetTickCount() - lastSysTime < MS2TICKS(sampleRate_ms)) {
			continue;
		}
		
		lastSysTime += MS2TICKS(sampleRate_ms);
		
		
		//Calculate averaged values
		float accels_avg_x = vtd->accels_data_sum_x / vtd->accels_sum_count;
		float accels_avg_y = vtd->accels_data_sum_y / vtd->accels_sum_count;
		float accels_avg_z = vtd->accels_data_sum_z / vtd->accels_sum_count;
		
		//Calculate DC bias
		float alpha=.005; //Hard-coded to drift very slowly
		vtd->accels_static_bias_x = alpha*accels_avg_x + (1-alpha)*vtd->accels_static_bias_x;
		vtd->accels_static_bias_y = alpha*accels_avg_y + (1-alpha)*vtd->accels_static_bias_y;
		vtd->accels_static_bias_z = alpha*accels_avg_z + (1-alpha)*vtd->accels_static_bias_z;
		
		// Add averaged values to the buffer, and remove DC bias. Only add real component, the
		// complex component was already set to zero by a memset operation
		vtd->accel_buffer_complex_x_q15[sample_count*2] = (accels_avg_x - vtd->accels_static_bias_x)*FLOAT_TO_Q15 + 0.5f; // Extra +0.5 rounds value when casting to int
		vtd->accel_buffer_complex_y_q15[sample_count*2] = (accels_avg_y - vtd->accels_static_bias_y)*FLOAT_TO_Q15 + 0.5f; // Extra +0.5 rounds value when casting to int
		vtd->accel_buffer_complex_z_q15[sample_count*2] = (accels_avg_z - vtd->accels_static_bias_z)*FLOAT_TO_Q15 + 0.5f; // Extra +0.5 rounds value when casting to int
		
		//Reset the accumulators
		vtd->accels_data_sum_x = 0;
		vtd->accels_data_sum_y = 0;
		vtd->accels_data_sum_z = 0;
		vtd->accels_sum_count = 0;

		// Advance sample and reset when at buffer end
		sample_count++;
		
		// Only process once the buffers are filled. This could be done continuously, 
		// but this way is probably easier on the processor
		if (sample_count >= vtd->fft_window_size) {
			//Reset sample count
			sample_count = 0;
			
			// Perform the DFT on each of the three axes
			for (int i=0; i < 3; i++) {
				if (status == ARM_MATH_SUCCESS) {
					
					//Create pointer and assign buffer vectors to it
					int16_t *ptr_cmplx_vec;
					
					switch (i) {
						case 0:
							ptr_cmplx_vec = vtd->accel_buffer_complex_x_q15;
							break;
						case 1:
							ptr_cmplx_vec = vtd->accel_buffer_complex_y_q15;
							break;
						case 2:
							ptr_cmplx_vec = vtd->accel_buffer_complex_z_q15;
							break;
						default:
							//Whoops, this is a major error, leave before we overwrite memory
							continue;
					}
					
					// Process the data through the CFFT/CIFFT module. This is an in-place
					// operation, so the FFT output is saved onto the accelerometer input buffer. 
					// Moving forward from this point, ptr_cmplx_vec contains the DFT of the
					// acceleration signal. 
					// While the input is Q15, the output is not (see next comment)
					arm_cfft_radix4_q15(&cfft_instance, ptr_cmplx_vec);
					
					// Upscale ptr_cmplx_vec back into Q15 format. The number of bits necessary is defined in 
					// ARM's arm_cfft_radix4_q15 documentation, figure CFFTQ15.gif
					arm_shift_q15(ptr_cmplx_vec, vtd->num_upscale_bits, ptr_cmplx_vec, vtd->fft_window_size);
					
					// Process the data through the Complex Magnitude Module. This calculates
					// the magnitude of each complex number, so that the output is a scalar
					// magnitude without complex phase. Only the first half of the values are
					// calculated because in a Fourier transform the second half is symmetric.
					arm_cmplx_mag_q15(ptr_cmplx_vec, vtd->fft_output, vtd->fft_window_size>>1);
					
					// Upscale fft_output back into Q15 format
					arm_shift_q15(vtd->fft_output, 1, vtd->fft_output, vtd->fft_window_size>>1);
					
					// Save RAM by copying back onto original input vector.
					memcpy(ptr_cmplx_vec, vtd->fft_output, (vtd->fft_window_size>>1) * sizeof(typeof(*vtd->accel_buffer_complex_x_q15)));
				}
			}
			
			//Write output to UAVO
			for (int j=0; j < (vtd->fft_window_size>>1); j++) 
			{
				//Assertion check that we are not trying to write to instances that don't exist
				if (j >= VibrationAnalysisOutputGetNumInstances())
					continue;
				
				vibrationAnalysisOutputData.x = vtd->accel_buffer_complex_x_q15[j]/FLOAT_TO_Q15;
				vibrationAnalysisOutputData.y = vtd->accel_buffer_complex_y_q15[j]/FLOAT_TO_Q15;
				vibrationAnalysisOutputData.z = vtd->accel_buffer_complex_z_q15[j]/FLOAT_TO_Q15;
				VibrationAnalysisOutputInstSet(j, &vibrationAnalysisOutputData);
			}
			
			
			// Erase buffer, which has the effect of setting the complex part to 0.
			memset(vtd->accel_buffer_complex_x_q15, 0, vtd->fft_window_size*2*sizeof(typeof(*(vtd->accel_buffer_complex_x_q15))));
			memset(vtd->accel_buffer_complex_y_q15, 0, vtd->fft_window_size*2*sizeof(typeof(*(vtd->accel_buffer_complex_y_q15))));
			memset(vtd->accel_buffer_complex_z_q15, 0, vtd->fft_window_size*2*sizeof(typeof(*(vtd->accel_buffer_complex_z_q15))));
		}
	}
}

/**
 * @}
 * @}
 */
