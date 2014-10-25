#include "pios_config.h"

/* C Lib Includes */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>

#include <stdint.h>
#include <stdbool.h>

#define NELEMENTS(x) (sizeof(x) / sizeof(*(x)))

#if defined(PIOS_INCLUDE_I2C)
#include <pios_i2c.h>
#endif

#if defined(PIOS_INCLUDE_FREERTOS)
#include "FreeRTOS_ut.h"
#endif
