/**
 ******************************************************************************
 * @addtogroup PIOS PIOS Core hardware abstraction layer
 * @{
 * @addtogroup   PIOS_I2C I2C Functions
 * @brief STM32 Hardware dependent I2C functionality
 * @{
 *
 * @file       pios_i2c.c
 * @author     The OpenPilot Team, http://www.openpilot.org Copyright (C) 2010.
 * @brief      I2C Enable/Disable routines
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

// Based on drv_i2c.c from baseflight - 32 bit fork of the MultiWii RC flight controller firmware

/*
 * == DO NOT USE FreeRTOS API in PIOS_I2C_EV_IRQ_Handler ==
 * 
 * Important STM32 Errata:
 *
 * See STM32F10xx8 and STM32F10xxB Errata sheet (Doc ID 14574 Rev 8),
 * Section 2.11.1, 2.11.2.
 *
 * 2.11.1:
 * When the EV7, EV7_1, EV6_1, EV6_3, EV2, EV8, and EV3 events are not
 * managed before the current byte is being transferred, problems may be
 * encountered such as receiving an extra byte, reading the same data twice
 * or missing data.
 *
 * 2.11.2:
 * In Master Receiver mode, when closing the communication using
 * method 2, the content of the last read data can be corrupted.
 *
 * If the user software is not able to read the data N-1 before the STOP
 * condition is generated on the bus, the content of the shift register
 * (data N) will be corrupted. (data N is shifted 1-bit to the left).
 *
 * ----------------------------------------------------------------------
 *
 * In order to ensure that events are not missed, the i2c interrupt must
 * not be preempted. We set the i2c interrupt priority to be the highest
 * interrupt in the system (priority level 0 / PIOS_IRQ_PRIO_EXTREME). 
 *
 * Since priority level 0 is above configMAX_SYSCALL_INTERRUPT_PRIORITY,
 * we cannot use any API functions in the I2C interrupt handlers (see FreeRTOS doc)
 *
 * As a result, we cannot use a semaphore (sem_ready) in PIOS_I2C_EV_IRQ_Handler
 */

/* Project Includes */
#include "pios.h"

#if defined(PIOS_INCLUDE_I2C)

//#define I2C_HALT_ON_ERRORS

#define I2C_DEFAULT_TIMEOUT 30000

#include <pios_i2c_priv.h>

#define I2C_CR1_FLAG_SWRST	((uint32_t)0x10008000)
#define I2C_CR1_FLAG_ALERT	((uint32_t)0x10002000)
#define I2C_CR1_FLAG_PEC	((uint32_t)0x10001000)
#define I2C_CR1_FLAG_POS	((uint32_t)0x10000800)
#define I2C_CR1_FLAG_ACK	((uint32_t)0x10000400)
#define I2C_CR1_FLAG_STOP	((uint32_t)0x10000200)
#define I2C_CR1_FLAG_START	((uint32_t)0x10000100)

struct i2c_internal_state {
    volatile uint16_t error_count;
    //--
    volatile bool busy;         // See sem_ready note above
    volatile uint8_t addr;
    volatile uint8_t reg;
    volatile uint8_t bytes;
    volatile uint8_t dir;
    volatile uint8_t* buf_p;    
    //--
    bool subaddress_sent;       // flag to indicate if subaddess sent
    bool final_stop;            // flag to indicate final bus condition
    int8_t index;               // index is signed -1 == send the subaddress
};

/*
 *
 *
 */
static void i2c_adapter_reset_bus(struct pios_i2c_adapter *i2c_adapter)
{
    GPIO_InitTypeDef scl_gpio_init;
    GPIO_InitTypeDef sda_gpio_init;

    scl_gpio_init = i2c_adapter->cfg->scl.init;
    scl_gpio_init.GPIO_Mode = GPIO_Mode_Out_OD;
    GPIO_Init(i2c_adapter->cfg->scl.gpio, &scl_gpio_init);

    sda_gpio_init = i2c_adapter->cfg->sda.init;
    sda_gpio_init.GPIO_Mode = GPIO_Mode_Out_OD;
    GPIO_Init(i2c_adapter->cfg->sda.gpio, &sda_gpio_init);

    GPIO_SetBits(i2c_adapter->cfg->scl.gpio, i2c_adapter->cfg->scl.init.GPIO_Pin);
    GPIO_SetBits(i2c_adapter->cfg->sda.gpio, i2c_adapter->cfg->sda.init.GPIO_Pin);

    for (uint8_t i = 0; i < 8; i++) {
        // Wait for any clock stretching to finish
        while (!GPIO_ReadInputDataBit(i2c_adapter->cfg->scl.gpio, i2c_adapter->cfg->scl.init.GPIO_Pin)) {
            PIOS_DELAY_WaituS(10);
        }

        // Pull low
        GPIO_ResetBits(i2c_adapter->cfg->scl.gpio, i2c_adapter->cfg->scl.init.GPIO_Pin); //Set bus low
        PIOS_DELAY_WaituS(10);
        // Release high again
        GPIO_SetBits(i2c_adapter->cfg->scl.gpio, i2c_adapter->cfg->scl.init.GPIO_Pin); //Set bus high
        PIOS_DELAY_WaituS(10);
    }

    // Generate a start then stop condition
    GPIO_ResetBits(i2c_adapter->cfg->sda.gpio, i2c_adapter->cfg->sda.init.GPIO_Pin); // Set bus data low
    PIOS_DELAY_WaituS(10);
    GPIO_ResetBits(i2c_adapter->cfg->scl.gpio, i2c_adapter->cfg->scl.init.GPIO_Pin); // Set bus scl low
    PIOS_DELAY_WaituS(10);
    GPIO_SetBits(i2c_adapter->cfg->scl.gpio, i2c_adapter->cfg->scl.init.GPIO_Pin); // Set bus scl high
    PIOS_DELAY_WaituS(10);
    GPIO_SetBits(i2c_adapter->cfg->sda.gpio, i2c_adapter->cfg->sda.init.GPIO_Pin); // Set bus sda high

    // Init pins
    GPIO_Init(i2c_adapter->cfg->scl.gpio, (GPIO_InitTypeDef*)&(i2c_adapter->cfg->scl.init));
    GPIO_Init(i2c_adapter->cfg->sda.gpio, (GPIO_InitTypeDef*)&(i2c_adapter->cfg->sda.init));
}

/*
 *
 *
 */
static void i2c_adapter_fsm_init(struct pios_i2c_adapter *i2c_adapter)
{
    // Init pins
    GPIO_Init(i2c_adapter->cfg->scl.gpio, (GPIO_InitTypeDef*)&(i2c_adapter->cfg->scl.init));
    GPIO_Init(i2c_adapter->cfg->sda.gpio, (GPIO_InitTypeDef*)&(i2c_adapter->cfg->sda.init));

    i2c_adapter_reset_bus(i2c_adapter);
    i2c_adapter->curr_state = I2C_STATE_STOPPED;

    // Reset the I2C block
    I2C_DeInit(i2c_adapter->cfg->regs);
    // Disable EVT and ERR interrupts - they are enabled by the first request
    I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_EVT | I2C_IT_ERR, DISABLE);
    // Initialize the I2C block
    I2C_Init(i2c_adapter->cfg->regs, (I2C_InitTypeDef*)&(i2c_adapter->cfg->init));
    // Enable I2C peripheral
    I2C_Cmd(i2c_adapter->cfg->regs, ENABLE);
}

/*
 *
 *
 */
static bool PIOS_I2C_validate(struct pios_i2c_adapter * i2c_adapter)
{
    return (i2c_adapter->magic == PIOS_I2C_DEV_MAGIC);
}

/*
 *
 *
 */
static struct pios_i2c_adapter * PIOS_I2C_alloc(void)
{
    struct pios_i2c_adapter * i2c_adapter;

    i2c_adapter = (struct pios_i2c_adapter *)PIOS_malloc(sizeof(*i2c_adapter) + 
        sizeof(struct i2c_internal_state));
    if (!i2c_adapter) {
        return(NULL);
    }

    i2c_adapter->magic = PIOS_I2C_DEV_MAGIC;
    i2c_adapter->active_txn = (const struct pios_i2c_txn *)(i2c_adapter+1);
    ((struct i2c_internal_state*)i2c_adapter->active_txn)->error_count = 0;
    return(i2c_adapter);
}

/**
* Initializes IIC driver
* \param[in] mode currently only mode 0 supported
* \return < 0 if initialisation failed
*/
int32_t PIOS_I2C_Init(uint32_t * i2c_id, const struct pios_i2c_adapter_cfg * cfg)
{
    PIOS_DEBUG_Assert(i2c_id);
    PIOS_DEBUG_Assert(cfg);

    struct pios_i2c_adapter * i2c_adapter;

    i2c_adapter = (struct pios_i2c_adapter *) PIOS_I2C_alloc();
    if (!i2c_adapter) {
        return(-1);
    }

    /* Bind the configuration to the device instance */
    i2c_adapter->cfg = cfg;

    i2c_adapter->sem_busy = PIOS_Semaphore_Create();
    i2c_adapter->sem_ready = NULL;

    /* Enable the associated peripheral clock */
    switch ((uint32_t) i2c_adapter->cfg->regs) {
    case (uint32_t) I2C1:
        /* Enable I2C peripheral clock (APB1 == slow speed) */
        RCC_APB1PeriphClockCmd(RCC_APB1Periph_I2C1, ENABLE);
        break;
    case (uint32_t) I2C2:
        /* Enable I2C peripheral clock (APB1 == slow speed) */
        RCC_APB1PeriphClockCmd(RCC_APB1Periph_I2C2, ENABLE);
        break;
    }

    if (i2c_adapter->cfg->remap) {
        GPIO_PinRemapConfig(i2c_adapter->cfg->remap, ENABLE);
    }

    /* Initialize the state machine */
    i2c_adapter_fsm_init(i2c_adapter);

    *i2c_id = (uint32_t)i2c_adapter;

    /* Configure and enable I2C interrupts */
    NVIC_Init((NVIC_InitTypeDef*)&(i2c_adapter->cfg->event.init));
    NVIC_Init((NVIC_InitTypeDef*)&(i2c_adapter->cfg->error.init));

    /* No error */
    return 0;
}

/**
 * @brief Perform a series of I2C transactions
 * @returns 0 if success or error code
 * @retval -1 for failed transaction
 * @retval -2 for failure to get semaphore
 */
int32_t PIOS_I2C_Transfer(uint32_t i2c_id, const struct pios_i2c_txn txn_list[], uint32_t num_txns)
{
    struct pios_i2c_adapter * i2c_adapter = (struct pios_i2c_adapter *)i2c_id;

    bool valid = PIOS_I2C_validate(i2c_adapter);
    PIOS_Assert(valid)

    struct i2c_internal_state* state = (struct i2c_internal_state*)i2c_adapter->active_txn;

    PIOS_DEBUG_Assert(txn_list);
    PIOS_DEBUG_Assert(num_txns);

    PIOS_DEBUG_Assert(num_txns == 1 || num_txns == 2);
    if (num_txns < 1 || num_txns > 2) {
        return -1;
    }

    if (PIOS_Semaphore_Take(i2c_adapter->sem_busy, i2c_adapter->cfg->transfer_timeout_ms) == false) {
#if defined(I2C_HALT_ON_ERRORS)
        PIOS_DEBUG_Assert(0);
#endif /*I2C_HALT_ON_ERRORS*/
        return -2;
     }

    i2c_adapter->bus_error = false;

    state->addr  = txn_list[0].addr << 1;
    state->reg   = txn_list[0].buf[0];

    // Reading
    if (num_txns == 2) {
        state->dir   = I2C_Direction_Receiver;
        state->buf_p = txn_list[1].buf;     // buf
        state->bytes = txn_list[1].len;    // len
    }
    // Writing
    else if (num_txns == 1) {
        state->dir   = I2C_Direction_Transmitter;
        state->buf_p = &(txn_list[0].buf[1]); // buf
        state->bytes = txn_list[0].len - 1;  // len
    }

    state->busy = true; 
 
    PIOS_DEBUG_Assert(!(i2c_adapter->cfg->regs->CR2 & I2C_IT_EVT));
    PIOS_DEBUG_Assert(!(i2c_adapter->cfg->regs->CR1 & 0x0100));

    // wait for any stop to finish sending
    while (i2c_adapter->cfg->regs->CR1 & 0x0200) { 
        PIOS_DELAY_WaituS(10);
    }

    // send the start for the new job
    I2C_GenerateSTART(i2c_adapter->cfg->regs, ENABLE);

    // allow the interrupts to fire off again
    I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_EVT | I2C_IT_ERR, ENABLE);

    uint32_t timeout = I2C_DEFAULT_TIMEOUT;
    while (state->busy && --timeout > 0);
    if (timeout == 0) {
        state->error_count++;
        // reinit peripheral + clock out garbage
        i2c_adapter_fsm_init(i2c_adapter);
#if defined(I2C_HALT_ON_ERRORS)
        PIOS_DEBUG_Assert(0);
#endif /*I2C_HALT_ON_ERRORS*/
    }

#if defined(I2C_HALT_ON_ERRORS)
    PIOS_DEBUG_Assert(!i2c_adapter->bus_error);
#endif

    PIOS_Semaphore_Give(i2c_adapter->sem_busy);

    return (timeout == 0) ? -2 :
       	i2c_adapter->bus_error ? -1 :
        0;
}

/**
 * @brief Check the I2C bus is clear and in a properly reset state
 * @returns  0 Bus is clear 
 * @returns -1 Bus is in use
 * @returns -2 Bus not clear
 */
int32_t PIOS_I2C_CheckClear(uint32_t i2c_id)
{
    struct pios_i2c_adapter * i2c_adapter = (struct pios_i2c_adapter *)i2c_id;

    bool valid = PIOS_I2C_validate(i2c_adapter);
    PIOS_Assert(valid)

    if (PIOS_Semaphore_Take(i2c_adapter->sem_busy, 0) == false)
            return -1;

    PIOS_Semaphore_Give(i2c_adapter->sem_busy);

    return 0;
}
/*
 *
 *
 */
void PIOS_I2C_EV_IRQ_Handler(uint32_t i2c_id)
{
    struct pios_i2c_adapter * i2c_adapter = (struct pios_i2c_adapter *)i2c_id;

    bool valid = PIOS_I2C_validate(i2c_adapter);
    PIOS_Assert(valid);

    struct i2c_internal_state* state = (struct i2c_internal_state*)i2c_adapter->active_txn;

    uint8_t SReg_1 = i2c_adapter->cfg->regs->SR1; //read the status register here

    if (SReg_1 & 0x0001) {      //we just sent a start - EV5 in ref manual
        i2c_adapter->cfg->regs->CR1 &= ~0x0800;  //reset the POS bit so ACK/NACK applied to the current byte
        I2C_AcknowledgeConfig(i2c_adapter->cfg->regs, ENABLE);    //make sure ACK is on
        state->index = 0;              //reset the index
        if ((state->dir == I2C_Direction_Receiver) && (state->subaddress_sent || 0xFF == state->reg)) {       //we have sent the subaddr
            state->subaddress_sent = 1;        //make sure this is set in case of no subaddress, so following code runs correctly
            if (state->bytes == 2)
                i2c_adapter->cfg->regs->CR1 |= 0x0800;    //set the POS bit so NACK applied to the final byte in the two byte read
            I2C_Send7bitAddress(i2c_adapter->cfg->regs, state->addr, I2C_Direction_Receiver);   //send the address and set hardware mode
        } else {                //direction is Tx, or we havent sent the sub and rep start
            I2C_Send7bitAddress(i2c_adapter->cfg->regs, state->addr, I2C_Direction_Transmitter);        //send the address and set hardware mode
            if (state->reg != 0xFF)       //0xFF as subaddress means it will be ignored, in Tx or Rx mode
                state->index = -1;     //send a subaddress
        }
    } else if (SReg_1 & 0x0002) {       //we just sent the address - EV6 in ref manual
        // Read SR1,2 to clear ADDR
        __DMB(); // memory fence to control hardware
        if (state->bytes == 1 && (state->dir == I2C_Direction_Receiver) && state->subaddress_sent) {     // we are receiving 1 byte - EV6_3
            I2C_AcknowledgeConfig(i2c_adapter->cfg->regs, DISABLE);           // turn off ACK
            __DMB();
            (void)i2c_adapter->cfg->regs->SR2;                                // clear ADDR after ACK is turned off
            I2C_GenerateSTOP(i2c_adapter->cfg->regs, ENABLE);                 // program the stop
            state->final_stop = true;
            I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_BUF, ENABLE);         // allow us to have an EV7
        } else {                    // EV6 and EV6_1
            (void)i2c_adapter->cfg->regs->SR2;        // clear the ADDR here
            __DMB();
            if (state->bytes == 2 && (state->dir == I2C_Direction_Receiver) && state->subaddress_sent) {     //rx 2 bytes - EV6_1
                I2C_AcknowledgeConfig(i2c_adapter->cfg->regs, DISABLE);   //turn off ACK
                I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_BUF, DISABLE);        //disable TXE to allow the buffer to fill
            } else if (state->bytes == 3 && (state->dir == I2C_Direction_Receiver) && state->subaddress_sent)       //rx 3 bytes
                I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_BUF, DISABLE);        //make sure RXNE disabled so we get a BTF in two bytes time
            else                //receiving greater than three bytes, sending subaddress, or transmitting
                I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_BUF, ENABLE);
        }
    } else if (SReg_1 & 0x004) {        //Byte transfer finished - EV7_2, EV7_3 or EV8_2
        state->final_stop = true;
        if ((state->dir == I2C_Direction_Receiver) && state->subaddress_sent) {     //EV7_2, EV7_3
            if (state->bytes > 2) {      //EV7_2
                I2C_AcknowledgeConfig(i2c_adapter->cfg->regs, DISABLE);   //turn off ACK
                state->buf_p[state->index++] = I2C_ReceiveData(i2c_adapter->cfg->regs);    //read data N-2
                I2C_GenerateSTOP(i2c_adapter->cfg->regs, ENABLE); //program the Stop
                state->final_stop = true; //reuired to fix hardware
                state->buf_p[state->index++] = I2C_ReceiveData(i2c_adapter->cfg->regs);    //read data N-1
                I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_BUF, ENABLE); //enable TXE to allow the final EV7
            } else {            //EV7_3
                if (state->final_stop)
                    I2C_GenerateSTOP(i2c_adapter->cfg->regs, ENABLE);     //program the Stop
                else
                    I2C_GenerateSTART(i2c_adapter->cfg->regs, ENABLE);    //program a rep start
                state->buf_p[state->index++] = I2C_ReceiveData(i2c_adapter->cfg->regs);    //read data N-1
                state->buf_p[state->index++] = I2C_ReceiveData(i2c_adapter->cfg->regs);    //read data N
                state->index++;        //to show job completed
            }
        } else {                //EV8_2, which may be due to a subaddress sent or a write completion
            if (state->subaddress_sent || ((state->dir == I2C_Direction_Transmitter))) {
                if (state->final_stop)
                    I2C_GenerateSTOP(i2c_adapter->cfg->regs, ENABLE);     //program the Stop
                else
                    I2C_GenerateSTART(i2c_adapter->cfg->regs, ENABLE);    //program a rep start
                state->index++;        //to show that the job is complete
            } else {            //We need to send a subaddress
                I2C_GenerateSTART(i2c_adapter->cfg->regs, ENABLE);        //program the repeated Start
                state->subaddress_sent = true;    //this is set back to zero upon completion of the current task
            }
        }
        //we must wait for the start to clear, otherwise we get constant BTF
        while (i2c_adapter->cfg->regs->CR1 & 0x0100) { ; }
    } else if (SReg_1 & 0x0040) {       //Byte received - EV7
        state->buf_p[state->index++] = I2C_ReceiveData(i2c_adapter->cfg->regs);
        if (state->bytes == (state->index + 3))
            I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_BUF, DISABLE);    //disable TXE to allow the buffer to flush so we can get an EV7_2
        if (state->bytes == state->index)       //We have completed a final EV7
            state->index++;            //to show job is complete
    } else if (SReg_1 & 0x0080) {       //Byte transmitted -EV8/EV8_1
        if (state->index != -1) {      //we dont have a subaddress to send
            I2C_SendData(i2c_adapter->cfg->regs, state->buf_p[state->index++]);
            if (state->bytes == state->index)   //we have sent all the data
                I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_BUF, DISABLE);        //disable TXE to allow the buffer to flush
        } else {
            state->index++;
            I2C_SendData(i2c_adapter->cfg->regs, state->reg);       //send the subaddress
            if ((state->dir == I2C_Direction_Receiver) || !state->bytes)      //if receiving or sending 0 bytes, flush now
                I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_BUF, DISABLE);        //disable TXE to allow the buffer to flush
        }
    }
    if (state->index == state->bytes + 1) {   //we have completed the current job
        //Completion Tasks go here
        //End of completion tasks
        state->subaddress_sent = false;    //reset this here
        // i2c_adapter->cfg->regs->CR1 &= ~0x0800;   //reset the POS bit so NACK applied to the current byte
        if (state->final_stop)  //If there is a final stop and no more jobs, bus is inactive, disable interrupts to prevent BTF
            I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_EVT | I2C_IT_ERR, DISABLE);       //Disable EVT and ERR interrupts while bus inactive
        state->busy = false;
    }
}

/*
 *
 *
 */
void PIOS_I2C_ER_IRQ_Handler(uint32_t i2c_id)
{
    struct pios_i2c_adapter * i2c_adapter = (struct pios_i2c_adapter *)i2c_id;

    bool valid = PIOS_I2C_validate(i2c_adapter);
    PIOS_Assert(valid);

    struct i2c_internal_state* state = (struct i2c_internal_state*)i2c_adapter->active_txn;

    volatile uint32_t SR1Register;
    // Read the I2C1 status register
    SR1Register = i2c_adapter->cfg->regs->SR1;
    if (SR1Register & 0x0F00) { //an error
        i2c_adapter->bus_error = true;
        // I2C1error.error = ((SR1Register & 0x0F00) >> 8);        //save error
        // I2C1error.job = job;    //the task
    }
    // If AF, BERR or ARLO, abandon the current job and commence new if there are jobs
    if (SR1Register & 0x0700) {
        (void)i2c_adapter->cfg->regs->SR2;                                // read second status register to clear ADDR if it is set (note that BTF will not be set after a NACK)
        I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_BUF, DISABLE);        // disable the RXNE/TXE interrupt - prevent the ISR tailchaining onto the ER (hopefully)
        if (!(SR1Register & 0x0200) && !(i2c_adapter->cfg->regs->CR1 & 0x0200)) {  // if we dont have an ARLO error, ensure sending of a stop
            if (i2c_adapter->cfg->regs->CR1 & 0x0100) {                   // We are currently trying to send a start, this is very bad as start,stop will hang the peripheral
                while (i2c_adapter->cfg->regs->CR1 & 0x0100);             // wait for any start to finish sending
                I2C_GenerateSTOP(i2c_adapter->cfg->regs, ENABLE);         // send stop to finalise bus transaction
                while (i2c_adapter->cfg->regs->CR1 & 0x0200);             // wait for stop to finish sending
                i2c_adapter_fsm_init(i2c_adapter);                        // reset and configure the hardware
            } else {
                I2C_GenerateSTOP(i2c_adapter->cfg->regs, ENABLE);         // stop to free up the bus
                I2C_ITConfig(i2c_adapter->cfg->regs, I2C_IT_EVT | I2C_IT_ERR, DISABLE);   // Disable EVT and ERR interrupts while bus inactive
            }
        }
    }
    i2c_adapter->cfg->regs->SR1 &= ~0x0F00;                               // reset all the error bits to clear the interrupt
    state->busy = false;
}

#endif

/**
  * @}
  * @}
  */
