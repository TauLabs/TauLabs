/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup PIOS_VIDEO Code for OSD video generator
 * @brief Output video (black & white pixels) over SPI
 * @{
 *
 * @file       pios_video.c
 * @author     Tau Labs, http://taulabs.org, Copyright (C) 2013-2014
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010-2014.
 * @brief      OSD gen module, handles OSD draw. Parts from CL-OSD and SUPEROSD projects
 * @see        The GNU Public License (GPL) Version 3
 *
 *****************************************************************************/
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#include "pios_config.h"

#if defined(PIOS_INCLUDE_VIDEO)

#if defined(PIOS_INCLUDE_FREERTOS)
#include "FreeRTOS.h"
#endif /* defined(PIOS_INCLUDE_FREERTOS) */

#include "pios.h"
#include "pios_video.h"
#include "pios_semaphore.h"

// How many frames until we redraw
#define VSYNC_REDRAW_CNT 1

extern struct pios_semaphore * onScreenDisplaySemaphore;

static const struct pios_video_type_boundary pios_video_type_boundary_ntsc = {
	.graphics_right  = 351,         // must be: graphics_width_real - 1
	.graphics_bottom = 239,         // must be: graphics_hight_real - 1
};

static const struct pios_video_type_boundary pios_video_type_boundary_pal = {
	.graphics_right  = 359,         // must be: graphics_width_real - 1
	.graphics_bottom = 265,         // must be: graphics_hight_real - 1
};

static const struct pios_video_type_cfg pios_video_type_cfg_ntsc = {
	.graphics_hight_real   = 240,   // Real visible lines
	.graphics_column_start = 130,   // First visible OSD column (after Hsync)
	.graphics_line_start   = 16,    // First visible OSD line
	.dma_buffer_length     = 45,    // DMA buffer length in bytes (graphics_right / 8 + 1)
	.period = 11,
	.dc     = (11 / 2),
};

static const struct pios_video_type_cfg pios_video_type_cfg_pal = {
	.graphics_hight_real   = 266,   // Real visible lines
	.graphics_column_start = 164,   // First visible OSD column (after Hsync)
	.graphics_line_start   = 25,    // First visible OSD line
	.dma_buffer_length     = 46,    // DMA buffer length in bytes ((graphics_right + 1) / 8 + 1)
	.period = 10,
	.dc     = (10 / 2),
};

// Allocate buffers.
// Must be allocated in one block, so it is in a struct.
struct _buffers {
	uint8_t buffer0_level[BUFFER_HEIGHT * BUFFER_WIDTH];
	uint8_t buffer0_mask[BUFFER_HEIGHT * BUFFER_WIDTH];
	uint8_t buffer1_level[BUFFER_HEIGHT * BUFFER_WIDTH];
	uint8_t buffer1_mask[BUFFER_HEIGHT * BUFFER_WIDTH];
} buffers;

// Remove the struct definition (makes it easier to write for).
#define buffer0_level (buffers.buffer0_level)
#define buffer0_mask  (buffers.buffer0_mask)
#define buffer1_level (buffers.buffer1_level)
#define buffer1_mask  (buffers.buffer1_mask)

// Pointers to each of these buffers.
uint8_t *draw_buffer_level;
uint8_t *draw_buffer_mask;
uint8_t *disp_buffer_level;
uint8_t *disp_buffer_mask;

volatile uint16_t active_line = 0;

const struct pios_video_type_boundary *pios_video_type_boundary_act = &pios_video_type_boundary_pal;

// Private variables
static int8_t x_offset = 0;
static int8_t x_offset_new = 0;
static int8_t y_offset = 0;
static const struct pios_video_cfg *dev_cfg = NULL;
static uint16_t num_video_lines = 0;
static int8_t video_type_tmp = VIDEO_TYPE_PAL;
static int8_t video_type_act = VIDEO_TYPE_NONE;
static const struct pios_video_type_cfg *pios_video_type_cfg_act = &pios_video_type_cfg_pal;

uint8_t black_pal = 30;
uint8_t white_pal = 110;
uint8_t black_ntsc = 10;
uint8_t white_ntsc = 110;

// Private functions
static void swap_buffers();
static void prepare_line();

/**
 * @brief Vsync interrupt service routine
 */
bool PIOS_Vsync_ISR()
{
#if defined(PIOS_LED_VIDEO)
	PIOS_LED_Toggle(PIOS_LED_VIDEO);
#endif

	static bool woken = false;
	static uint16_t Vsync_update = 0;

	// discard spurious vsync pulses (due to improper grounding), so we don't overload the CPU
	if (active_line > 0 && active_line < pios_video_type_cfg_ntsc.graphics_hight_real - 10) {
		return false;
	}

	// Stop the line counter
	TIM_Cmd(dev_cfg->line_counter, DISABLE);

	// Update the number of video lines
	num_video_lines = dev_cfg->line_counter->CNT;

	// check video type
	if (num_video_lines > VIDEO_TYPE_PAL_ROWS) {
		video_type_tmp = VIDEO_TYPE_PAL;
	}

	// if video type has changed set new active values
	if (video_type_act != video_type_tmp) {
		video_type_act = video_type_tmp;
		if (video_type_act == VIDEO_TYPE_NTSC) {
			pios_video_type_boundary_act = &pios_video_type_boundary_ntsc;
			pios_video_type_cfg_act = &pios_video_type_cfg_ntsc;
			dev_cfg->set_bw_levels(black_ntsc, white_ntsc);
		} else {
			pios_video_type_boundary_act = &pios_video_type_boundary_pal;
			pios_video_type_cfg_act = &pios_video_type_cfg_pal;
			dev_cfg->set_bw_levels(black_pal, white_pal);
		}

		switch(dev_cfg->pixel_timer.timer_chan) {
		case TIM_Channel_1:
			dev_cfg->pixel_timer.timer->CCR1 = pios_video_type_cfg_act->dc;
			break;
		case TIM_Channel_2:
			dev_cfg->pixel_timer.timer->CCR2 = pios_video_type_cfg_act->dc;
			break;
		case TIM_Channel_3:
			dev_cfg->pixel_timer.timer->CCR3 = pios_video_type_cfg_act->dc;
			break;
		case TIM_Channel_4:
			dev_cfg->pixel_timer.timer->CCR4 = pios_video_type_cfg_act->dc;
			break;
        }

		dev_cfg->pixel_timer.timer->ARR  = pios_video_type_cfg_act->period;
		dev_cfg->hsync_capture.timer->ARR = pios_video_type_cfg_act->dc * (pios_video_type_cfg_act->graphics_column_start + x_offset);
	}
	if (x_offset != x_offset_new)
	{
		x_offset = x_offset_new;
		dev_cfg->hsync_capture.timer->ARR = pios_video_type_cfg_act->dc * (pios_video_type_cfg_act->graphics_column_start + x_offset);
	}

	video_type_tmp = VIDEO_TYPE_NTSC;

	// Every VSYNC_REDRAW_CNT field: swap buffers and trigger redraw
	if (++Vsync_update >= VSYNC_REDRAW_CNT) {
		Vsync_update = 0;
		swap_buffers();
		PIOS_Semaphore_Give_FromISR(onScreenDisplaySemaphore, &woken);
	}

	// Get ready for the first line
	active_line = 0;

	// Set the number of lines to wait until we start clocking out pixels
	dev_cfg->line_counter->CNT = 0xffff - (pios_video_type_cfg_act->graphics_line_start + y_offset);
	TIM_Cmd(dev_cfg->line_counter, ENABLE);
#if defined(PIOS_INCLUDE_FREERTOS)
	/* Yield From ISR if needed */
	portEND_SWITCHING_ISR(woken == true ? pdTRUE : pdFALSE);
#endif
	return woken;
}


void PIOS_First_Line_ISR(void);
#if defined(PIOS_VIDEO_TIM4_COUNTER)
void TIM4_IRQHandler(void) __attribute__((alias("PIOS_First_Line_ISR")));
#endif /* defined(PIOS_VIDEO_TIM4_COUNTER) */

/**
 * ISR Triggered by line_counter, starts clocking out pixels for first visible OSD line
 */
void PIOS_First_Line_ISR(void)
{
	if(TIM_GetITStatus(dev_cfg->line_counter, TIM_IT_Update) && (active_line == 0))
	{
		// Clear the interrupt flag
		dev_cfg->line_counter->SR &= ~TIM_SR_UIF;

		// Prepare the first line
		prepare_line();

		// Hack: The timing for the first line is critical, so we output it again
		active_line = 0;

		// Get ready to count the remaining lines
		dev_cfg->line_counter->CNT = pios_video_type_cfg_act->graphics_line_start + y_offset;
		TIM_Cmd(dev_cfg->line_counter, ENABLE);
	}
}


void PIOS_VIDEO_DMA_Handler(void);
void DMA2_Stream3_IRQHandler(void) __attribute__((alias("PIOS_VIDEO_DMA_Handler")));
void DMA1_Stream4_IRQHandler(void) __attribute__((alias("PIOS_VIDEO_DMA_Handler")));

/**
 * DMA transfer complete interrupt handler
 * Note: This function is called for every line (~13k times / s), so we use direct register access for
 * efficiency
 */
void PIOS_VIDEO_DMA_Handler(void)
{	
	// Handle flags from DMA stream channel
	if ((dev_cfg->mask_dma->LISR & DMA_FLAG_TCIF3) && (dev_cfg->level_dma->HISR & DMA_FLAG_TCIF4)) {

		// Flush the SPI
		while ((dev_cfg->level.regs->SR & SPI_I2S_FLAG_TXE) == 0) {
			;
		}
		while (dev_cfg->level.regs->SR & SPI_I2S_FLAG_BSY) {
			;
		}
		while ((dev_cfg->mask.regs->SR & SPI_I2S_FLAG_TXE) == 0) {
			;
		}
		while (dev_cfg->mask.regs->SR & SPI_I2S_FLAG_BSY) {
			;
		}

		// Disable the SPI, makes sure the pins are LOW
		dev_cfg->mask.regs->CR1 &= (uint16_t)~SPI_CR1_SPE;
		dev_cfg->level.regs->CR1 &= (uint16_t)~SPI_CR1_SPE;

		if (active_line < pios_video_type_cfg_act->graphics_hight_real) { // lines existing
			prepare_line();
		} else { // last line completed
			// Clear the DMA interrupt flags
			dev_cfg->mask_dma->LIFCR  |= DMA_FLAG_TCIF3;
			dev_cfg->level_dma->HIFCR |= DMA_FLAG_TCIF4;

			// Stop pixel timer
			dev_cfg->pixel_timer.timer->CR1  &= (uint16_t) ~TIM_CR1_CEN;

			// Disable the pixel timer slave mode configuration
			dev_cfg->pixel_timer.timer->SMCR &= (uint16_t) ~TIM_SMCR_SMS;
			// Stop DMA
			dev_cfg->mask.dma.tx.channel->CR  &= ~(uint32_t)DMA_SxCR_EN;
			dev_cfg->level.dma.tx.channel->CR &= ~(uint32_t)DMA_SxCR_EN;
		}
	}
}

/**
 * Prepare the system to watch for a Hsync pulse to trigger the pixel clock and clock out the next line
 * Note: This function is called for every line (~13k times / s), so we use direct register access for
 * efficiency
 */
static inline void prepare_line()
{
	uint32_t buf_offset = active_line * BUFFER_WIDTH;

	// Prepare next line DMA:
	// Clear DMA interrupt flags
	dev_cfg->mask_dma->LIFCR  |= DMA_FLAG_TCIF3 | DMA_FLAG_HTIF3 | DMA_FLAG_FEIF3 | DMA_FLAG_TEIF3 | DMA_FLAG_DMEIF3;
	dev_cfg->level_dma->HIFCR |= DMA_FLAG_TCIF4 | DMA_FLAG_HTIF4 | DMA_FLAG_FEIF4 | DMA_FLAG_TEIF4 | DMA_FLAG_DMEIF4;
	// Load new line
	dev_cfg->mask.dma.tx.channel->M0AR  = (uint32_t)&disp_buffer_mask[buf_offset];
	dev_cfg->level.dma.tx.channel->M0AR = (uint32_t)&disp_buffer_level[buf_offset];
	// Set length
	dev_cfg->mask.dma.tx.channel->NDTR  = (uint16_t)pios_video_type_cfg_act->dma_buffer_length;
	dev_cfg->level.dma.tx.channel->NDTR = (uint16_t)pios_video_type_cfg_act->dma_buffer_length;

	// Stop pixel timer
	dev_cfg->pixel_timer.timer->CR1  &= (uint16_t) ~TIM_CR1_CEN;

	// Set initial value
	dev_cfg->pixel_timer.timer->CNT   = 0;

	// Reset the SMS bits
	dev_cfg->pixel_timer.timer->SMCR &= (uint16_t) ~TIM_SMCR_SMS;
	dev_cfg->pixel_timer.timer->SMCR |= TIM_SlaveMode_Trigger;

	// Enable SPI
	dev_cfg->mask.regs->CR1  |= SPI_CR1_SPE;
	dev_cfg->level.regs->CR1 |= SPI_CR1_SPE;

	// Enable DMA
	dev_cfg->mask.dma.tx.channel->CR  |= (uint32_t)DMA_SxCR_EN;
	dev_cfg->level.dma.tx.channel->CR |= (uint32_t)DMA_SxCR_EN;

	// Advance line counter
	active_line++;
}


/**
 * swap_buffers: Swaps the two buffers. Contents in the display
 * buffer is seen on the output and the display buffer becomes
 * the new draw buffer.
 */
static void swap_buffers()
{
	// While we could use XOR swap this is more reliable and
	// dependable and it's only called a few times per second.
	// Many compilers should optimize these to EXCH instructions.
	uint8_t *tmp;

	SWAP_BUFFS(tmp, disp_buffer_mask, draw_buffer_mask);
	SWAP_BUFFS(tmp, disp_buffer_level, draw_buffer_level);
}


/**
 * Init
 */
void PIOS_Video_Init(const struct pios_video_cfg *cfg)
{
	TIM_TimeBaseInitTypeDef  TIM_TimeBaseStructure;
	dev_cfg = cfg; // store config before enabling interrupt

	// This code currently only works with SPI1 and SPI2, catch misconfigurations here
	if ((cfg->mask.regs != SPI1) && (cfg->mask.regs != SPI2))
		PIOS_Assert(0);

	if ((cfg->level.regs != SPI1) && (cfg->level.regs != SPI2))
		PIOS_Assert(0);

	// SPI for maskbuffer
	GPIO_Init(cfg->mask.sclk.gpio, (GPIO_InitTypeDef *)&(cfg->mask.sclk.init));
	GPIO_Init(cfg->mask.miso.gpio, (GPIO_InitTypeDef *)&(cfg->mask.miso.init));
	if (cfg->mask.remap) {
		GPIO_PinAFConfig(cfg->mask.sclk.gpio, __builtin_ctz(cfg->mask.sclk.init.GPIO_Pin), cfg->mask.remap);
		GPIO_PinAFConfig(cfg->mask.miso.gpio, __builtin_ctz(cfg->mask.miso.init.GPIO_Pin), cfg->mask.remap);
	}

	// SPI for levelbuffer
	GPIO_Init(cfg->level.sclk.gpio, (GPIO_InitTypeDef *)&(cfg->level.sclk.init));
	GPIO_Init(cfg->level.miso.gpio, (GPIO_InitTypeDef *)&(cfg->level.miso.init));
	if (cfg->level.remap) {
		GPIO_PinAFConfig(cfg->level.sclk.gpio, __builtin_ctz(cfg->level.sclk.init.GPIO_Pin), cfg->level.remap);
		GPIO_PinAFConfig(cfg->level.miso.gpio, __builtin_ctz(cfg->level.miso.init.GPIO_Pin), cfg->level.remap);
	}

	// HSYNC captrue timer: Start counting at HSYNC and start pixel timer after at correct x-position
	GPIO_Init(cfg->hsync_capture.pin.gpio, (GPIO_InitTypeDef *)&(cfg->hsync_capture.pin.init));
	if (cfg->hsync_capture.remap) {
		GPIO_PinAFConfig(cfg->hsync_capture.pin.gpio, __builtin_ctz(cfg->hsync_capture.pin.init.GPIO_Pin), cfg->hsync_capture.remap);
	}

	TIM_TimeBaseStructInit(&TIM_TimeBaseStructure);
	TIM_TimeBaseStructure.TIM_Period = pios_video_type_cfg_act->dc * (pios_video_type_cfg_act->graphics_column_start + x_offset);
	TIM_TimeBaseStructure.TIM_Prescaler = 0;
	TIM_TimeBaseStructure.TIM_ClockDivision = 0;
	TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
	TIM_TimeBaseInit(cfg->hsync_capture.timer, &TIM_TimeBaseStructure);

	TIM_SelectOnePulseMode(cfg->hsync_capture.timer, TIM_OPMode_Single);
	TIM_SelectSlaveMode(cfg->hsync_capture.timer, TIM_SlaveMode_Trigger);
	TIM_SelectInputTrigger(cfg->hsync_capture.timer, TIM_TS_TI2FP2);

	TIM_SelectMasterSlaveMode(cfg->hsync_capture.timer, TIM_MasterSlaveMode_Enable);
	TIM_SelectOutputTrigger(cfg->hsync_capture.timer, TIM_TRGOSource_Update);

	// Pixel timer: Outputs clock for SPI
	GPIO_Init(cfg->pixel_timer.pin.gpio, (GPIO_InitTypeDef *)&(cfg->pixel_timer.pin.init));
	if (cfg->pixel_timer.remap) {
		GPIO_PinAFConfig(cfg->pixel_timer.pin.gpio, __builtin_ctz(cfg->pixel_timer.pin.init.GPIO_Pin), cfg->pixel_timer.remap);
	}

	switch(cfg->pixel_timer.timer_chan) {
	case TIM_Channel_1:
		TIM_OC1Init(cfg->pixel_timer.timer, (TIM_OCInitTypeDef*)&cfg->tim_oc_init);
		TIM_OC1PreloadConfig(cfg->pixel_timer.timer, TIM_OCPreload_Enable);
		TIM_SetCompare1(cfg->pixel_timer.timer, pios_video_type_cfg_act->dc);
		break;
	case TIM_Channel_2:
		TIM_OC2Init(cfg->pixel_timer.timer, (TIM_OCInitTypeDef*)&cfg->tim_oc_init);
		TIM_OC2PreloadConfig(cfg->pixel_timer.timer, TIM_OCPreload_Enable);
		TIM_SetCompare2(cfg->pixel_timer.timer, pios_video_type_cfg_act->dc);
		break;
	case TIM_Channel_3:
		TIM_OC3Init(cfg->pixel_timer.timer, (TIM_OCInitTypeDef*)&cfg->tim_oc_init);
		TIM_OC3PreloadConfig(cfg->pixel_timer.timer, TIM_OCPreload_Enable);
		TIM_SetCompare3(cfg->pixel_timer.timer, pios_video_type_cfg_act->dc);
		break;
	case TIM_Channel_4:
		TIM_OC4Init(cfg->pixel_timer.timer, (TIM_OCInitTypeDef*)&cfg->tim_oc_init);
		TIM_OC4PreloadConfig(cfg->pixel_timer.timer, TIM_OCPreload_Enable);
		TIM_SetCompare4(cfg->pixel_timer.timer, pios_video_type_cfg_act->dc);
		break;
	}

	TIM_SetAutoreload(cfg->pixel_timer.timer, pios_video_type_cfg_act->period);
	TIM_ARRPreloadConfig(cfg->pixel_timer.timer, ENABLE);
	TIM_CtrlPWMOutputs(cfg->pixel_timer.timer, ENABLE);

	if (cfg->hsync_capture.timer == TIM2)
		TIM_SelectInputTrigger(cfg->pixel_timer.timer, TIM_TS_ITR1);
	else
		PIOS_Assert(0);

	// Line counter: Counts number of HSYNCS (from hsync_capture) and triggers output of first visible line
	TIM_TimeBaseStructInit(&TIM_TimeBaseStructure);
	TIM_TimeBaseStructure.TIM_Period = 0xffff;
	TIM_TimeBaseStructure.TIM_Prescaler = 0;
	TIM_TimeBaseStructure.TIM_ClockDivision = 0;
	TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
	TIM_TimeBaseInit(cfg->line_counter, &TIM_TimeBaseStructure);

	/* Enable the TIM4 gloabal Interrupt */
	NVIC_InitTypeDef NVIC_InitStructure;

	if (cfg->line_counter == TIM4)
		NVIC_InitStructure.NVIC_IRQChannel = TIM4_IRQn;
	else
		PIOS_Assert(0);

	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = PIOS_IRQ_PRIO_HIGHEST;
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 1;
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
	NVIC_Init(&NVIC_InitStructure);

	/* The line_counter counts the trigger output from hsync_capture */
	if (cfg->hsync_capture.timer == TIM2)
		TIM_SelectInputTrigger(cfg->line_counter, TIM_TS_ITR1);
	else
		PIOS_Assert(0);

	TIM_SelectSlaveMode(cfg->line_counter, TIM_SlaveMode_External1);
	TIM_SelectOnePulseMode(cfg->line_counter, TIM_OPMode_Single);
	TIM_ITConfig(cfg->line_counter, TIM_IT_CC1 | TIM_IT_CC2 | TIM_IT_CC3 | TIM_IT_CC4 | TIM_IT_COM | TIM_IT_Trigger | TIM_IT_Break, DISABLE);
	TIM_Cmd(cfg->line_counter, DISABLE);


	/* Initialize the SPI block */
	SPI_Init(cfg->level.regs, (SPI_InitTypeDef *)&(cfg->level.init));
	SPI_Init(cfg->mask.regs, (SPI_InitTypeDef *)&(cfg->mask.init));

	/* Configure DMA for SPI*/
	DMA_Init(cfg->mask.dma.tx.channel, (DMA_InitTypeDef *)&(cfg->mask.dma.tx.init));
	DMA_Init(cfg->level.dma.tx.channel, (DMA_InitTypeDef *)&(cfg->level.dma.tx.init));

	/* Trigger interrupt when transfer complete */
	DMA_ITConfig(cfg->level.dma.tx.channel, DMA_IT_TC, ENABLE);
	DMA_ITConfig(cfg->mask.dma.tx.channel, DMA_IT_TC, ENABLE);

	/* Configure and clear buffers */
	draw_buffer_level = buffer0_level;
	draw_buffer_mask  = buffer0_mask;
	disp_buffer_level = buffer1_level;
	disp_buffer_mask  = buffer1_mask;
	memset(disp_buffer_mask, 0, BUFFER_HEIGHT * BUFFER_WIDTH);
	memset(disp_buffer_level, 0, BUFFER_HEIGHT * BUFFER_WIDTH);
	memset(draw_buffer_mask, 0, BUFFER_HEIGHT * BUFFER_WIDTH);
	memset(draw_buffer_level, 0, BUFFER_HEIGHT * BUFFER_WIDTH);

	/* Configure DMA interrupt */
	NVIC_Init((NVIC_InitTypeDef*)&cfg->level.dma.irq.init);
	NVIC_Init((NVIC_InitTypeDef*)&cfg->mask.dma.irq.init);

	/* Enable SPI interrupts to DMA */
	SPI_I2S_DMACmd(cfg->mask.regs, SPI_I2S_DMAReq_Tx, ENABLE);
	SPI_I2S_DMACmd(cfg->level.regs, SPI_I2S_DMAReq_Tx, ENABLE);



	// Enable interrupts
	PIOS_EXTI_Init(cfg->vsync);
	TIM_ITConfig(cfg->line_counter, TIM_IT_Update, ENABLE);

	// Enable the capture timer
	TIM_Cmd(cfg->hsync_capture.timer, ENABLE);
}

/**
 *
 */
uint16_t PIOS_Video_GetLines(void)
{
	return num_video_lines;
}

/**
 *
 */
uint16_t PIOS_Video_GetType(void)
{
	return video_type_act;
}

/**
*  Set the black and white levels
*/
void PIOS_Video_SetLevels(uint8_t black_pal_in, uint8_t white_pal_in, uint8_t black_ntsc_in, uint8_t white_ntsc_in)
{
	if (video_type_act == VIDEO_TYPE_PAL)
		dev_cfg->set_bw_levels(black_pal_in, white_pal_in);
	else
		dev_cfg->set_bw_levels(black_ntsc_in, white_ntsc_in);
	black_pal = black_pal_in;
	white_pal = white_pal_in;
	black_ntsc = black_ntsc_in;
	white_ntsc = white_ntsc_in;
}

/**
*  Set the offset in x direction
*/
void PIOS_Video_SetXOffset(int8_t x_offset_in)
{
	if (x_offset_in > 20)
		x_offset_in = 20;
	if (x_offset_in < -20)
		x_offset_in = -20;

	x_offset_new = x_offset_in;
	//dev_cfg->hsync_capture.timer->ARR = pios_video_type_cfg_act->dc * (pios_video_type_cfg_act->graphics_column_start + x_offset);
}

/**
*  Set the offset in y direction
*/
void PIOS_Video_SetYOffset(int8_t y_offset_in)
{
	if (y_offset_in > 20)
		y_offset_in = 20;
	if (y_offset_in < -20)
		y_offset_in = -20;
	y_offset = y_offset_in;
}

#endif /* PIOS_INCLUDE_VIDEO */
