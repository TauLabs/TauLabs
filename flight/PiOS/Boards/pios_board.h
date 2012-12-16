#ifndef PIOS_BOARD_H_
#define PIOS_BOARD_H_

#ifdef USE_STM32103CB_PIPXTREME
#include "STM32103CB_PIPXTREME_Rev1.h"
#elif USE_STM32103CB_CC_Rev1
#include "STM32103CB_CC_Rev1.h"
#elif USE_STM32F4xx_OP
#include "STM32F4xx_Revolution.h"
#elif USE_STM32F4xx_OSD
#include "STM32F4xx_OSD.h"
#elif USE_STM32F4xx_RM
#include "STM32F4xx_RevoMini.h"
#elif USE_STM32F4xx_FREEDOM
#include "STM32F4xx_Freedom.h"
#elif USE_STM32F4xx_QUANTON
#include "STM32F4xx_Quanton.h"
#elif USE_STM32F4xx_FLYINGF4
#include "STM32F4xx_FlyingF4.h"
#elif USE_STM32F4xx_DISCOVERYF4
#include "STM32F4xx_DiscoveryF4.h"
#else
#error Board definition has not been provided.
#endif

#endif /* PIOS_BOARD_H_ */
