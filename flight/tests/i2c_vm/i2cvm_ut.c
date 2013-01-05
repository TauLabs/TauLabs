#include "i2cvm.h"
#include <string.h>		/* memcpy */

I2CVMData uavo_data;

void I2CVMSet(I2CVMData * data)
{
	/* Grab a snapshot of the UAVO data contents */
	memcpy(&uavo_data, data, sizeof(uavo_data));

	return;
}

