/*
 * realtime.c
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
#include "realtime_dt.h"

/* options for Simulink Desktop Real-Time board 0 */
static double SLDRTBoardOptions0[] = {
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
};

/* list of Simulink Desktop Real-Time timers */
const int SLDRTTimerCount = 1;
const double SLDRTTimers[2] = {
  0.04, 0.0,
};

/* list of Simulink Desktop Real-Time boards */
const int SLDRTBoardCount = 1;
SLDRTBOARD SLDRTBoards[1] = {
  { "National_Instruments/PCI-6221", 4294967295U, 5, SLDRTBoardOptions0 },
};

/* Block signals (auto storage) */
B_realtime_T realtime_B;

/* Block states (auto storage) */
DW_realtime_T realtime_DW;

/* Real-time model */
RT_MODEL_realtime_T realtime_M_;
RT_MODEL_realtime_T *const realtime_M = &realtime_M_;

/* Model output function */
void realtime_output(void)
{
  real_T currentTime;

  /* S-Function (sldrtai): '<Root>/Analog Input1' */
  /* S-Function Block: <Root>/Analog Input1 */
  {
    ANALOGIOPARM parm;
    parm.mode = (RANGEMODE) realtime_P.AnalogInput1_RangeMode;
    parm.rangeidx = realtime_P.AnalogInput1_VoltRange;
    RTBIO_DriverIO(0, ANALOGINPUT, IOREAD, 1, &realtime_P.AnalogInput1_Channels,
                   &realtime_B.AnalogInput1, &parm);
  }

  /* Step: '<Root>/Step1' */
  currentTime = realtime_M->Timing.t[0];
  if (currentTime < realtime_P.Step1_Time) {
    realtime_B.Step1 = realtime_P.Step1_Y0;
  } else {
    realtime_B.Step1 = realtime_P.Step1_YFinal;
  }

  /* End of Step: '<Root>/Step1' */

  /* S-Function (sldrtao): '<Root>/Analog Output1' */
  /* S-Function Block: <Root>/Analog Output1 */
  {
    {
      ANALOGIOPARM parm;
      parm.mode = (RANGEMODE) realtime_P.AnalogOutput1_RangeMode;
      parm.rangeidx = realtime_P.AnalogOutput1_VoltRange;
      RTBIO_DriverIO(0, ANALOGOUTPUT, IOWRITE, 1,
                     &realtime_P.AnalogOutput1_Channels, ((real_T*)
        (&realtime_B.Step1)), &parm);
    }
  }
}

/* Model update function */
void realtime_update(void)
{
  /* Update absolute time for base rate */
  /* The "clockTick0" counts the number of times the code of this task has
   * been executed. The absolute time is the multiplication of "clockTick0"
   * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
   * overflow during the application lifespan selected.
   * Timer of this task consists of two 32 bit unsigned integers.
   * The two integers represent the low bits Timing.clockTick0 and the high bits
   * Timing.clockTickH0. When the low bit overflows to 0, the high bits increment.
   */
  if (!(++realtime_M->Timing.clockTick0)) {
    ++realtime_M->Timing.clockTickH0;
  }

  realtime_M->Timing.t[0] = realtime_M->Timing.clockTick0 *
    realtime_M->Timing.stepSize0 + realtime_M->Timing.clockTickH0 *
    realtime_M->Timing.stepSize0 * 4294967296.0;

  {
    /* Update absolute timer for sample time: [0.04s, 0.0s] */
    /* The "clockTick1" counts the number of times the code of this task has
     * been executed. The absolute time is the multiplication of "clockTick1"
     * and "Timing.stepSize1". Size of "clockTick1" ensures timer will not
     * overflow during the application lifespan selected.
     * Timer of this task consists of two 32 bit unsigned integers.
     * The two integers represent the low bits Timing.clockTick1 and the high bits
     * Timing.clockTickH1. When the low bit overflows to 0, the high bits increment.
     */
    if (!(++realtime_M->Timing.clockTick1)) {
      ++realtime_M->Timing.clockTickH1;
    }

    realtime_M->Timing.t[1] = realtime_M->Timing.clockTick1 *
      realtime_M->Timing.stepSize1 + realtime_M->Timing.clockTickH1 *
      realtime_M->Timing.stepSize1 * 4294967296.0;
  }
}

/* Model initialize function */
void realtime_initialize(void)
{
  /* Start for S-Function (sldrtao): '<Root>/Analog Output1' */

  /* S-Function Block: <Root>/Analog Output1 */
  {
    {
      ANALOGIOPARM parm;
      parm.mode = (RANGEMODE) realtime_P.AnalogOutput1_RangeMode;
      parm.rangeidx = realtime_P.AnalogOutput1_VoltRange;
      RTBIO_DriverIO(0, ANALOGOUTPUT, IOWRITE, 1,
                     &realtime_P.AnalogOutput1_Channels,
                     &realtime_P.AnalogOutput1_InitialValue, &parm);
    }
  }
}

/* Model terminate function */
void realtime_terminate(void)
{
  /* Terminate for S-Function (sldrtao): '<Root>/Analog Output1' */

  /* S-Function Block: <Root>/Analog Output1 */
  {
    {
      ANALOGIOPARM parm;
      parm.mode = (RANGEMODE) realtime_P.AnalogOutput1_RangeMode;
      parm.rangeidx = realtime_P.AnalogOutput1_VoltRange;
      RTBIO_DriverIO(0, ANALOGOUTPUT, IOWRITE, 1,
                     &realtime_P.AnalogOutput1_Channels,
                     &realtime_P.AnalogOutput1_FinalValue, &parm);
    }
  }
}

/*========================================================================*
 * Start of Classic call interface                                        *
 *========================================================================*/
void MdlOutputs(int_T tid)
{
  realtime_output();
  UNUSED_PARAMETER(tid);
}

void MdlUpdate(int_T tid)
{
  realtime_update();
  UNUSED_PARAMETER(tid);
}

void MdlInitializeSizes(void)
{
}

void MdlInitializeSampleTimes(void)
{
}

void MdlInitialize(void)
{
}

void MdlStart(void)
{
  realtime_initialize();
}

void MdlTerminate(void)
{
  realtime_terminate();
}

/* Registration function */
RT_MODEL_realtime_T *realtime(void)
{
  /* Registration code */

  /* initialize non-finites */
  rt_InitInfAndNaN(sizeof(real_T));

  /* initialize real-time model */
  (void) memset((void *)realtime_M, 0,
                sizeof(RT_MODEL_realtime_T));

  {
    /* Setup solver object */
    rtsiSetSimTimeStepPtr(&realtime_M->solverInfo,
                          &realtime_M->Timing.simTimeStep);
    rtsiSetTPtr(&realtime_M->solverInfo, &rtmGetTPtr(realtime_M));
    rtsiSetStepSizePtr(&realtime_M->solverInfo, &realtime_M->Timing.stepSize0);
    rtsiSetErrorStatusPtr(&realtime_M->solverInfo, (&rtmGetErrorStatus
      (realtime_M)));
    rtsiSetRTModelPtr(&realtime_M->solverInfo, realtime_M);
  }

  rtsiSetSimTimeStep(&realtime_M->solverInfo, MAJOR_TIME_STEP);
  rtsiSetSolverName(&realtime_M->solverInfo,"FixedStepDiscrete");

  /* Initialize timing info */
  {
    int_T *mdlTsMap = realtime_M->Timing.sampleTimeTaskIDArray;
    mdlTsMap[0] = 0;
    mdlTsMap[1] = 1;
    realtime_M->Timing.sampleTimeTaskIDPtr = (&mdlTsMap[0]);
    realtime_M->Timing.sampleTimes = (&realtime_M->Timing.sampleTimesArray[0]);
    realtime_M->Timing.offsetTimes = (&realtime_M->Timing.offsetTimesArray[0]);

    /* task periods */
    realtime_M->Timing.sampleTimes[0] = (0.0);
    realtime_M->Timing.sampleTimes[1] = (0.04);

    /* task offsets */
    realtime_M->Timing.offsetTimes[0] = (0.0);
    realtime_M->Timing.offsetTimes[1] = (0.0);
  }

  rtmSetTPtr(realtime_M, &realtime_M->Timing.tArray[0]);

  {
    int_T *mdlSampleHits = realtime_M->Timing.sampleHitArray;
    mdlSampleHits[0] = 1;
    mdlSampleHits[1] = 1;
    realtime_M->Timing.sampleHits = (&mdlSampleHits[0]);
  }

  rtmSetTFinal(realtime_M, 20.0);
  realtime_M->Timing.stepSize0 = 0.04;
  realtime_M->Timing.stepSize1 = 0.04;

  /* External mode info */
  realtime_M->Sizes.checksums[0] = (3019373800U);
  realtime_M->Sizes.checksums[1] = (3719948379U);
  realtime_M->Sizes.checksums[2] = (3470267410U);
  realtime_M->Sizes.checksums[3] = (1659599792U);

  {
    static const sysRanDType rtAlwaysEnabled = SUBSYS_RAN_BC_ENABLE;
    static RTWExtModeInfo rt_ExtModeInfo;
    static const sysRanDType *systemRan[1];
    realtime_M->extModeInfo = (&rt_ExtModeInfo);
    rteiSetSubSystemActiveVectorAddresses(&rt_ExtModeInfo, systemRan);
    systemRan[0] = &rtAlwaysEnabled;
    rteiSetModelMappingInfoPtr(realtime_M->extModeInfo,
      &realtime_M->SpecialInfo.mappingInfo);
    rteiSetChecksumsPtr(realtime_M->extModeInfo, realtime_M->Sizes.checksums);
    rteiSetTPtr(realtime_M->extModeInfo, rtmGetTPtr(realtime_M));
  }

  realtime_M->solverInfoPtr = (&realtime_M->solverInfo);
  realtime_M->Timing.stepSize = (0.04);
  rtsiSetFixedStepSize(&realtime_M->solverInfo, 0.04);
  rtsiSetSolverMode(&realtime_M->solverInfo, SOLVER_MODE_SINGLETASKING);

  /* block I/O */
  realtime_M->blockIO = ((void *) &realtime_B);

  {
    realtime_B.AnalogInput1 = 0.0;
    realtime_B.Step1 = 0.0;
  }

  /* parameters */
  realtime_M->defaultParam = ((real_T *)&realtime_P);

  /* states (dwork) */
  realtime_M->dwork = ((void *) &realtime_DW);
  (void) memset((void *)&realtime_DW, 0,
                sizeof(DW_realtime_T));

  /* data type transition information */
  {
    static DataTypeTransInfo dtInfo;
    (void) memset((char_T *) &dtInfo, 0,
                  sizeof(dtInfo));
    realtime_M->SpecialInfo.mappingInfo = (&dtInfo);
    dtInfo.numDataTypes = 14;
    dtInfo.dataTypeSizes = &rtDataTypeSizes[0];
    dtInfo.dataTypeNames = &rtDataTypeNames[0];

    /* Block I/O transition table */
    dtInfo.BTransTable = &rtBTransTable;

    /* Parameters transition table */
    dtInfo.PTransTable = &rtPTransTable;
  }

  /* Initialize Sizes */
  realtime_M->Sizes.numContStates = (0);/* Number of continuous states */
  realtime_M->Sizes.numY = (0);        /* Number of model outputs */
  realtime_M->Sizes.numU = (0);        /* Number of model inputs */
  realtime_M->Sizes.sysDirFeedThru = (0);/* The model is not direct feedthrough */
  realtime_M->Sizes.numSampTimes = (2);/* Number of sample times */
  realtime_M->Sizes.numBlocks = (4);   /* Number of blocks */
  realtime_M->Sizes.numBlockIO = (2);  /* Number of block outputs */
  realtime_M->Sizes.numBlockPrms = (15);/* Sum of parameter "widths" */
  return realtime_M;
}

/*========================================================================*
 * End of Classic call interface                                          *
 *========================================================================*/
