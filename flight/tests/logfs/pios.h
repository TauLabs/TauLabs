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

#include <pios_heap.h>

// portTICK_RATE_MS is in [ms/tick].
// See http://sourceforge.net/tracker/?func=detail&aid=3498382&group_id=111543&atid=659636
#define TICKS2MS(t)	((t) * (portTICK_RATE_MS))
#define MS2TICKS(m)	((m) / (portTICK_RATE_MS))

/* Would be from pios_debug.h but that file pulls on way too many dependencies */
#define PIOS_Assert(x) if (!(x)) { while (1) ; }
#define PIOS_DEBUG_Assert(x) PIOS_Assert(x)
