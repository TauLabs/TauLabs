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

// portTICK_RATE_MS is in [ms/tick].
// See http://sourceforge.net/tracker/?func=detail&aid=3498382&group_id=111543&atid=659636
#define TICKS2MS(t)	((t) * (portTICK_RATE_MS))
#define MS2TICKS(m)	((m) / (portTICK_RATE_MS))

#if defined(PIOS_INCLUDE_I2C)
#include <pios_i2c.h>
#endif

#if defined(PIOS_INCLUDE_FREERTOS)
#include "FreeRTOS_ut.h"
#endif
