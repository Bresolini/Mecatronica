/*
 * realtime_data.c
 *
 * Code generation for model "realtime".
 *
 * Model version              : 1.26
 * Simulink Coder version : 8.12 (R2017a) 16-Feb-2017
 * C source code generated on : Tue Aug 27 17:48:48 2019
 *
 * Target selection: sldrt.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#include "realtime.h"
#include "realtime_private.h"

/* Block parameters (auto storage) */
P_realtime_T realtime_P = {
  0.0,                                 /* Mask Parameter: AnalogOutput1_FinalValue
                                        * Referenced by: '<Root>/Analog Output1'
                                        */
  0.0,                                 /* Mask Parameter: AnalogOutput1_InitialValue
                                        * Referenced by: '<Root>/Analog Output1'
                                        */
  20.0,                                /* Mask Parameter: AnalogInput1_MaxMissedTicks
                                        * Referenced by: '<Root>/Analog Input1'
                                        */
  20.0,                                /* Mask Parameter: AnalogOutput1_MaxMissedTicks
                                        * Referenced by: '<Root>/Analog Output1'
                                        */
  0.0,                                 /* Mask Parameter: AnalogInput1_YieldWhenWaiting
                                        * Referenced by: '<Root>/Analog Input1'
                                        */
  0.0,                                 /* Mask Parameter: AnalogOutput1_YieldWhenWaiting
                                        * Referenced by: '<Root>/Analog Output1'
                                        */
  4,                                   /* Mask Parameter: AnalogInput1_Channels
                                        * Referenced by: '<Root>/Analog Input1'
                                        */
  1,                                   /* Mask Parameter: AnalogOutput1_Channels
                                        * Referenced by: '<Root>/Analog Output1'
                                        */
  0,                                   /* Mask Parameter: AnalogInput1_RangeMode
                                        * Referenced by: '<Root>/Analog Input1'
                                        */
  0,                                   /* Mask Parameter: AnalogOutput1_RangeMode
                                        * Referenced by: '<Root>/Analog Output1'
                                        */
  0,                                   /* Mask Parameter: AnalogInput1_VoltRange
                                        * Referenced by: '<Root>/Analog Input1'
                                        */
  0,                                   /* Mask Parameter: AnalogOutput1_VoltRange
                                        * Referenced by: '<Root>/Analog Output1'
                                        */
  1.0,                                 /* Expression: 1
                                        * Referenced by: '<Root>/Step1'
                                        */
  0.0,                                 /* Expression: 0
                                        * Referenced by: '<Root>/Step1'
                                        */
  3.0                                  /* Expression: 3
                                        * Referenced by: '<Root>/Step1'
                                        */
};
