/* ----------------------------------------------------------------------------    
* Copyright (C) 2010 ARM Limited. All rights reserved.    
*    
* $Date:        15. February 2012  
* $Revision: 	V1.1.0  
*    
* Project: 	    CMSIS DSP Library    
* Title:		arm_q15_to_q31.c    
*    
* Description:	Converts the elements of the Q15 vector to Q31 vector.  
*    
* Target Processor: Cortex-M4/Cortex-M3/Cortex-M0
*  
* Version 1.1.0 2012/02/15 
*    Updated with more optimizations, bug fixes and minor API changes.  
*   
* Version 1.0.10 2011/7/15  
*    Big Endian support added and Merged M0 and M3/M4 Source code.   
*    
* Version 1.0.3 2010/11/29   
*    Re-organized the CMSIS folders and updated documentation.    
*     
* Version 1.0.2 2010/11/11    
*    Documentation updated.     
*    
* Version 1.0.1 2010/10/05     
*    Production release and review comments incorporated.    
*    
* Version 1.0.0 2010/09/20     
*    Production release and review comments incorporated.    
* ---------------------------------------------------------------------------- */

#include "arm_math.h"

/**    
 * @ingroup groupSupport    
 */

/**    
 * @addtogroup q15_to_x    
 * @{    
 */

/**    
 * @brief Converts the elements of the Q15 vector to Q31 vector.     
 * @param[in]       *pSrc points to the Q15 input vector    
 * @param[out]      *pDst points to the Q31 output vector   
 * @param[in]       blockSize length of the input vector    
 * @return none.    
 *    
 * \par Description:    
 *    
 * The equation used for the conversion process is:   
 *   
 * <pre>    
 * 	pDst[n] = (q31_t) pSrc[n] << 16;   0 <= n < blockSize.    
 * </pre>    
 *   
 */


void arm_q15_to_q31(
  q15_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize)
{
  q15_t *pIn = pSrc;                             /* Src pointer */
  uint32_t blkCnt;                               /* loop counter */

#ifndef ARM_MATH_CM0

  /* Run the below code for Cortex-M4 and Cortex-M3 */
  q31_t in1, in2;
  q31_t out1, out2, out3, out4;

  /*loop Unrolling */
  blkCnt = blockSize >> 2u;

  /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.    
   ** a second loop below computes the remaining 1 to 3 samples. */
  while(blkCnt > 0u)
  {
    /* C = (q31_t)A << 16 */
    /* convert from q15 to q31 and then store the results in the destination buffer */
    in1 = *__SIMD32(pIn)++;
    in2 = *__SIMD32(pIn)++;

#ifndef ARM_MATH_BIG_ENDIAN

    /* extract lower 16 bits to 32 bit result */
    out1 = in1 << 16u;
    /* extract upper 16 bits to 32 bit result */
    out2 = in1 & 0xFFFF0000;
    /* extract lower 16 bits to 32 bit result */
    out3 = in2 << 16u;
    /* extract upper 16 bits to 32 bit result */
    out4 = in2 & 0xFFFF0000;

#else

    /* extract upper 16 bits to 32 bit result */
    out1 = in1 & 0xFFFF0000;
    /* extract lower 16 bits to 32 bit result */
    out2 = in1 << 16u;
    /* extract upper 16 bits to 32 bit result */
    out3 = in2 & 0xFFFF0000;
    /* extract lower 16 bits to 32 bit result */
    out4 = in2 << 16u;

#endif //      #ifndef ARM_MATH_BIG_ENDIAN

    *pDst++ = out1;
    *pDst++ = out2;
    *pDst++ = out3;
    *pDst++ = out4;

    /* Decrement the loop counter */
    blkCnt--;
  }

  /* If the blockSize is not a multiple of 4, compute any remaining output samples here.    
   ** No loop unrolling is used. */
  blkCnt = blockSize % 0x4u;

#else

  /* Run the below code for Cortex-M0 */

  /* Loop over blockSize number of values */
  blkCnt = blockSize;

#endif /* #ifndef ARM_MATH_CM0 */

  while(blkCnt > 0u)
  {
    /* C = (q31_t)A << 16 */
    /* convert from q15 to q31 and then store the results in the destination buffer */
    *pDst++ = (q31_t) * pIn++ << 16;

    /* Decrement the loop counter */
    blkCnt--;
  }

}

/**    
 * @} end of q15_to_x group    
 */
