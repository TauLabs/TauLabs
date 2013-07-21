/* PIOS Feature Selection */
#include "pios_config.h"

#if defined(PIOS_INCLUDE_FREERTOS)
/* FreeRTOS Includes */
#include "FreeRTOS.h"
#endif

#if defined(PIOS_INCLUDE_FLASH)
#include <pios_flash.h>
#include <pios_flashfs.h>
#endif

#define TICKS2MS(t)	((t) * portTICK_RATE_MS) // portTICK_RATE_MS is in [ms/tick]. It is poorly named, see
#define MS2TICKS(m)	((m) / portTICK_RATE_MS) // http://sourceforge.net/tracker/?func=detail&aid=3498382&group_id=111543&atid=659636

/* Would be from pios_debug.h but that file pulls on way too many dependencies */
#define PIOS_Assert(x) if (!(x)) { while (1) ; }
#define PIOS_DEBUG_Assert(x) PIOS_Assert(x)
