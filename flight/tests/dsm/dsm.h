
#include "stdint.h"
#include "stdbool.h"

#define PIOS_DSM_NUM_INPUTS     12
#define DSM_CHANNELS_PER_FRAME  7
#define DSM_FRAME_LENGTH        (1+1+DSM_CHANNELS_PER_FRAME*2)
#define DSM_DSM2_RES_MASK       0x0010
#define DSM_2ND_FRAME_MASK      0x8000

enum pios_dsm_dev_magic {
	PIOS_DSM_DEV_MAGIC = 0x44534d78,
};

enum dsm_resolution {
	DSM_UNKNOWN, DSM_10BIT, DSM_11BIT
};

struct pios_dsm_state {
	uint16_t channel_data[PIOS_DSM_NUM_INPUTS];
	uint8_t received_data[DSM_FRAME_LENGTH];
	uint8_t receive_timer;
	uint8_t failsafe_timer;
	uint8_t frame_found;
	uint8_t byte_count;
#ifdef DSM_LOST_FRAME_COUNTER
	uint8_t	frames_lost_last;
	uint16_t frames_lost;
#endif
};

struct pios_dsm_dev {
	enum pios_dsm_dev_magic magic;
	const struct pios_dsm_cfg *cfg;
	struct pios_dsm_state state;
	enum dsm_resolution resolution;
};

int PIOS_DSM_Reset(struct pios_dsm_dev *dsm_dev);
int PIOS_DSM_UnrollChannels(struct pios_dsm_dev *dsm_dev);
int PIOS_DSM_GetResolution(struct pios_dsm_dev *dsm_dev);