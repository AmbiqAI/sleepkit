
#include "arm_math.h"
#include <stdbool.h>

#include "ns_ambiqsuite_harness.h"

#include "constants.h"
#include "stimulus.h"
#include "physiokit.h"
#include "sleepkit.h"
#include "model.h"

#define MAX_RR_PEAKS (4*SK_DATA_LEN/SAMPLE_RATE)
#define VERBOSE 1

arm_biquad_casd_df1_inst_f32 ppg1SosInst;
arm_biquad_casd_df1_inst_f32 ppg2SosInst;
arm_biquad_casd_df1_inst_f32 ecgSosInst;
arm_biquad_casd_df1_inst_f32 qrsSosInst;

static float32_t pkArena[5*SK_DATA_LEN];

static float32_t spo2Coefs[3] = {1.5958422, -34.6596622, 112.6898759};

static float32_t ppg1Clean[SK_DATA_LEN];
static float32_t ppg2Clean[SK_DATA_LEN];
static float32_t ecgClean[SK_DATA_LEN];
static float32_t qrsClean[SK_DATA_LEN];

static uint32_t ppg1Peaks[MAX_RR_PEAKS];
static uint32_t ppg2Peaks[MAX_RR_PEAKS];
static uint32_t qrsPeaks[MAX_RR_PEAKS];

static uint32_t ppg1RRIntervals[MAX_RR_PEAKS];
static uint32_t ppg2RRIntervals[MAX_RR_PEAKS];
static uint32_t ecgRRIntervals[MAX_RR_PEAKS];

static uint8_t ppg1Mask[SK_DATA_LEN];
static uint8_t ppg2Mask[SK_DATA_LEN];
static uint8_t ecgMask[SK_DATA_LEN];

static hrv_td_metrics_t ppg1Hrv;
static hrv_td_metrics_t ppg2Hrv;
static hrv_td_metrics_t ecgHrv;

// print(generate_arm_biquad_sos(0.5, 30, 250, order=3, var_name="ecgBandPass"))
#define ECG_SOS_LEN (3)
static float32_t ecgSosState[4 * ECG_SOS_LEN];
static float32_t ecgSos[5 * ECG_SOS_LEN] = {
    0.027461107467472153, 0.054922214934944306, 0.027461107467472153, 1.0997280329991979, -0.5012977494269584,
    1.0, 0.0, -1.0, 1.433070925390203, -0.44021887101067064,
    1.0, -2.0, 1.0, 1.9875591985256609, -0.987718549527938
};

// print(generate_arm_biquad_sos(0.7, 4, 250, order=3, var_name="ppgBandPass"))
#define PPG_SOS_LEN (3)
static float32_t ppg1SosState[4 * PPG_SOS_LEN];
static float32_t ppg2SosState[4 * PPG_SOS_LEN];
static float32_t ppgSos[5 * PPG_SOS_LEN] = {
   7.806553586407756e-05, 0.00015613107172815513, 7.806553586407756e-05, 1.9166292410880583, -0.9255546729411119,
   1.0, 0.0, -1.0, 1.914478508417793, -0.9156892158611577,
   1.0, -2.0, 1.0, 1.9893385827438674, -0.989509721800255
};

// print(generate_arm_biquad_sos(10, 30, 250, order=3, var_name="qrsBandPass"))
#define QRS_SOS_LEN (3)
static float32_t qrsSosState[4 * QRS_SOS_LEN];
static float32_t qrsSos[5 * QRS_SOS_LEN] = {
   0.01018257673643693, 0.02036515347287386, 0.01018257673643693, 1.439786547927553, -0.5913983513994713,
   1.0, 0.0, -1.0, 1.2823329392866671, -0.7030006898068755,
   1.0, -2.0, 1.0, 1.8049684603984224, -0.8702175029734399
};


static biquad_filt_f32_t ecgFilter = {
    .inst = &ecgSosInst,
    .numSecs = ECG_SOS_LEN,
    .sos = ecgSos,
    .state = ecgSosState
};

static biquad_filt_f32_t qrsFilter = {
    .inst = &qrsSosInst,
    .numSecs = QRS_SOS_LEN,
    .sos = qrsSos,
    .state = qrsSosState
};

static biquad_filt_f32_t ppg1Filter = {
    .inst = &ppg1SosInst,
    .numSecs = PPG_SOS_LEN,
    .sos = ppgSos,
    .state = ppg1SosState
};

static biquad_filt_f32_t ppg2Filter = {
    .inst = &ppg2SosInst,
    .numSecs = PPG_SOS_LEN,
    .sos = ppgSos,
    .state = ppg2SosState
};

static ppg_peak_f32_t ppgFindPeakCtx = {
    .peakWin=0.111,
    .beatWin=0.666,
    .beatOffset=0.333,
    .peakDelayWin=0.3,
    .sampleRate=SAMPLE_RATE,
    .state=pkArena
};

static ecg_peak_f32_t qrsFindPeakCtx = {
    .qrsWin=0.1,
    .avgWin=1.0,
    .qrsPromWeight=1.5,
    .qrsMinLenWeight=0.4,
    .qrsDelayWin=0.3,
    .sampleRate=SAMPLE_RATE,
    .state=pkArena
};

void
print_array_f32(float32_t *arr, size_t len, char *name) {
    ns_printf("%s = np.array([", name);
    for (size_t i = 0; i < len; i++) {
        ns_printf("%f, ", arr[i]);
    }
    ns_printf("])\n");
}

void
print_array_u32(uint32_t *arr, size_t len, char *name) {
    ns_printf("%s = np.array([", name);
    for (size_t i = 0; i < len; i++) {
        ns_printf("%lu, ", arr[i]);
    }
    ns_printf("])\n");
}

uint32_t
init_sleepkit() {
    uint32_t err = 0;
    pk_init_biquad_filter(&ppg1Filter);
    pk_init_biquad_filter(&ppg2Filter);
    pk_init_biquad_filter(&ecgFilter);
    pk_init_biquad_filter(&qrsFilter);
    return err;
}


uint32_t
sk_preprocess(float32_t *ppg1Data, float32_t *ppg2Data, float32_t *ecgData, size_t dataLen) {
    /**
     * @brief Preprocess signals
     *
     */
    uint32_t err = 0;
    float32_t spo2, hr;
    float32_t ppg1Mean, ppg2Mean, ecgMean;
    uint32_t numPpg1Peaks, numPpg2Peaks, numQrsPeaks;

    // 1. Extract mean of PPG signals (needed for SpO2 calculation)
    err |= pk_mean(ppg1Data, &ppg1Mean, dataLen);
    err |= pk_mean(ppg2Data, &ppg2Mean, dataLen);
    err |= pk_mean(ecgData, &ecgMean, dataLen);
#ifdef VERBOSE
    ns_printf("PPG1 mean: %f\n", ppg1Mean);
    ns_printf("PPG2 mean: %f\n", ppg2Mean);
    ns_printf(" ECG mean: %f\n", ecgMean);
    ns_printf("\n\n");
#endif

    // 2. Filter PPG/ECG signals using biquad filter
    arm_offset_f32(ppg1Data, -ppg1Mean, ppg1Data, dataLen);
    arm_offset_f32(ppg2Data, -ppg2Mean, ppg2Data, dataLen);
    arm_offset_f32(ecgData, -ecgMean, ecgData, dataLen);

    err |= pk_apply_biquad_filtfilt(&ppg1Filter, ppg1Data, ppg1Clean, dataLen, pkArena);
    err |= pk_apply_biquad_filtfilt(&ppg2Filter, ppg2Data, ppg2Clean, dataLen, pkArena);
    err |= pk_apply_biquad_filtfilt(&ecgFilter, ecgData, ecgClean, dataLen, pkArena);
    err |= pk_apply_biquad_filtfilt(&qrsFilter, ecgData, qrsClean, dataLen, pkArena);

#ifdef VERBOSE
    print_array_f32(ppg1Clean, dataLen, "ppg1_clean");
    print_array_f32(ppg2Clean, dataLen, "ppg2_clean");
    print_array_f32(qrsClean, dataLen, "qrs_clean");
    print_array_f32(ecgClean, dataLen, "ecg_clean");
#endif

    // 3. Find peaks in PPG/ECG signals
    numPpg1Peaks = pk_ppg_find_peaks(&ppgFindPeakCtx, ppg1Clean, dataLen, ppg1Peaks);
    numPpg2Peaks = pk_ppg_find_peaks(&ppgFindPeakCtx, ppg2Clean, dataLen, ppg2Peaks);
    numQrsPeaks = pk_ecg_find_peaks(&qrsFindPeakCtx, qrsClean, dataLen, qrsPeaks);

#ifdef VERBOSE
    print_array_u32(ppg1Peaks, numPpg1Peaks, "ppg1_peaks");
    print_array_u32(ppg2Peaks, numPpg2Peaks, "ppg2_peaks");
    print_array_u32(qrsPeaks, numQrsPeaks, "ecg_peaks");
#endif

    // 4. Compute RR intervals from peaks
    pk_compute_rr_intervals(ppg1Peaks, numPpg1Peaks, ppg1RRIntervals);
    pk_compute_rr_intervals(ppg2Peaks, numPpg2Peaks, ppg2RRIntervals);
    pk_compute_rr_intervals(qrsPeaks, numQrsPeaks, ecgRRIntervals);

    // 5. Filter PPG/ECG peaks and RR intervals
    pk_filter_rr_intervals(ppg1RRIntervals, numPpg1Peaks, ppg1Mask, SAMPLE_RATE);
    pk_filter_rr_intervals(ppg2RRIntervals, numPpg2Peaks, ppg2Mask, SAMPLE_RATE);
    pk_filter_rr_intervals(ecgRRIntervals, numQrsPeaks, ecgMask, SAMPLE_RATE);

#ifdef VERBOSE
    print_array_u32(ppg1RRIntervals, numPpg1Peaks, "ppg1_rr_ints");
    print_array_u32(ppg2RRIntervals, numPpg2Peaks, "ppg2_rr_ints");
    print_array_u32(ecgRRIntervals, numQrsPeaks, "ecg_rr_ints");
#endif

    // 6. Compute SpO2 from PPG signals (AC/DC ratio)
    spo2 = pk_compute_spo2_in_time(ppg1Clean, ppg2Clean, ppg1Mean, ppg2Mean, SK_DATA_LEN, spo2Coefs, SAMPLE_RATE);

    // 7. Compute HRV from RR intervals
    pk_compute_hrv_from_rr_intervals(ecgRRIntervals, numQrsPeaks, ecgMask, &ecgHrv);

    hr = 60/(ecgHrv.meanNN/SAMPLE_RATE);
    ns_printf("SpO2: %0.2f\n", spo2);
    ns_printf("HR: %0.2f\n", hr);
    ns_printf("sdNN: %0.2f\n", ecgHrv.sdNN);
    ns_printf("nn20: %lu nn50: %lu\n", ecgHrv.nn20, ecgHrv.nn50);
    ns_printf("sdSD: %0.2f rmsSD: %0.2f\n", ecgHrv.sdSD, ecgHrv.rmsSD);
    ns_printf("min: %0.2f max: %0.2f\n", ecgHrv.minNN, ecgHrv.maxNN);
    return err;
}
