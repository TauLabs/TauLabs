
#include "dsm.h"
#include "stdio.h"

int PIOS_DSM_Reset(struct pios_dsm_dev *dsm_dev)
{
	dsm_dev->resolution = DSM_UNKNOWN;

	for (int i = 0; i < PIOS_DSM_NUM_INPUTS; i++) {
		dsm_dev->state.channel_data[i] = 0;
	}

	return 0;
}

int PIOS_DSM_GetResolution(struct pios_dsm_dev *dsm_dev)
{
	uint8_t resolution = (dsm_dev->resolution == DSM_10BIT) ? 10 : 11;
	return resolution;
}

/**
 * DSM packets expect to have sequential channel numbers but
 * based on resolution they will be shifted by one position
 */
enum dsm_resolution PIOS_DSM_DetectResolution(uint8_t *packet)
{
	uint8_t channel0, channel1;
	uint16_t word0, word1;
	bool bit_10, bit_11;

	uint8_t *s = &packet[2];

	// Check for 10 bit
	word0 = ((uint16_t)s[0] << 8) | s[1];
	word1 = ((uint16_t)s[2] << 8) | s[3];

	// Don't detect on the second data packets
	if (word0 & DSM_2ND_FRAME_MASK)
		return DSM_UNKNOWN;

	channel0 = (word0 >> 10) & 0x0f;
	channel1 = (word1 >> 10) & 0x0f;
	bit_10 = (channel0 == 1) && (channel1 == 5);

	// Check for 11 bit
	channel0 = (word0 >> 11) & 0x0f;
	channel1 = (word1 >> 11) & 0x0f;
	bit_11 = (channel0 == 1) && (channel1 == 5);

	if (bit_10 && !bit_11)
		return DSM_10BIT;
	if (bit_11 && !bit_10)
		return DSM_11BIT;
	return DSM_UNKNOWN;
}

/**
 * This is the code from the PIOS_DSM layer
 */
int PIOS_DSM_UnrollChannels(struct pios_dsm_dev *dsm_dev)
{
	struct pios_dsm_state *state = &(dsm_dev->state);
	/* Fix resolution for detection. */

#ifdef DSM_LOST_FRAME_COUNTER
	/* increment the lost frame counter */
	uint8_t frames_lost = state->received_data[0];
	state->frames_lost += (frames_lost - state->frames_lost_last);
	state->frames_lost_last = frames_lost;
#endif

	// If no stream type has yet been detected, then try to probe for it
	// this should only happen once per power cycle
	if (dsm_dev->resolution == DSM_UNKNOWN) {
		dsm_dev->resolution = PIOS_DSM_DetectResolution(state->received_data);
	}

	/* Stream type still not detected */
	if (dsm_dev->resolution == DSM_UNKNOWN) {
		return -2;
	}
	uint8_t resolution = (dsm_dev->resolution == DSM_10BIT) ? 10 : 11;
	uint16_t mask = (dsm_dev->resolution == DSM_10BIT) ? 0x03ff : 0x07ff;

	/* unroll channels */
	uint8_t *s = &(state->received_data[2]);

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
		state->channel_data[channel_num] = (word & mask);
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