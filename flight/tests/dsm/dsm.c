
#include "dsm.h"
#include "stdio.h"


static uint8_t resolution = 11;
static uint32_t depth = 0;

int PIOS_DSM_Reset()
{
	resolution = 11;
	depth = 0;
	return 0;
}

int PIOS_DSM_GetResolution()
{
	return resolution;
}

/**
 * This is the code from the PIOS_DSM layer
 */
int PIOS_DSM_UnrollChannels(struct pios_dsm_state *state)
{
	/* Fix resolution for detection. */
	uint32_t channel_log = 0;

#ifdef DSM_LOST_FRAME_COUNTER
	/* increment the lost frame counter */
	uint8_t frames_lost = state->received_data[0];
	state->frames_lost += (frames_lost - state->frames_lost_last);
	state->frames_lost_last = frames_lost;
#endif

	/* unroll channels */
	uint8_t *s = &(state->received_data[2]);
	uint16_t mask = (resolution == 10) ? 0x03ff : 0x07ff;

	for (int i = 0; i < DSM_CHANNELS_PER_FRAME; i++) {
		uint16_t word = ((uint16_t)s[0] << 8) | s[1];
		s += 2;

		/* skip empty channel slot */
		if (word == 0xffff)
			continue;

		/* minimal data validation */
		if ((i > 0) && (word & DSM_2ND_FRAME_MASK)) {
			/* invalid frame data, ignore rest of the frame */
			goto stream_error;
		}

		/* extract and save the channel value */
		uint8_t channel_num = (word >> resolution) & 0x0f;
		if (channel_num < PIOS_DSM_NUM_INPUTS) {
			if (channel_log & (1 << channel_num)) {
				/* Found duplicate! */
				/* Update resolution and restart processing the current frame. */
				resolution = 10;
				return PIOS_DSM_UnrollChannels(state);
			}
			state->channel_data[channel_num] = (word & mask);
			/* keep track of this channel */
			channel_log |= (1 << channel_num);
		}
	}
#ifdef DSM_LOST_FRAME_COUNTER
	/* put lost frames counter into the last channel for debugging */
	state->channel_data[PIOS_DSM_NUM_INPUTS-1] = state->frames_lost;
#endif

	/* all channels processed */
	return 0;

stream_error:
	/* either DSM2 selected with DSMX stream found, or vice-versa */
	return -1;
}