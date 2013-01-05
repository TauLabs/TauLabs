#include <stdint.h>

#define I2CVM_RAM_NUMELEMENTS 8

typedef struct {
	uint8_t ram[I2CVM_RAM_NUMELEMENTS];

	uint16_t pc;

	int32_t r0;
	int32_t r1;
	int32_t r2;
	int32_t r3;
	int32_t r4;
	int32_t r5;
	int32_t r6;
} I2CVMData;

extern void I2CVMSet(I2CVMData * data);

/* Window into the latest UAVO contents */
extern I2CVMData uavo_data;
