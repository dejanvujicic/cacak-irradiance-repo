/*
 * Edge-deployment benchmark for the Sensors manuscript, Section 4.8.
 */

#include <Arduino.h>
#include <math.h>
#include "edge_models.h"
#include "test_vectors.h"

#define REPS 1000
#define BATCH 20   

/* ---------------- XGBoost ---------------- */
static inline float xgb_tree(const int16_t *F, const float *T,
                             const uint16_t *Y, const uint16_t *N,
                             uint16_t off, const float *x) {
  uint16_t n = off;
  while (F[n] >= 0) n = off + ((x[F[n]] < T[n]) ? Y[n] : N[n]);
  return T[n];
}
static float xgb_predict(int target, const float *x) {
  float s = xgb_base;
  if (target == 0)
    for (int t = 0; t < XGB_T0_NTREES; t++)
      s += xgb_tree(xgb_t0_feat, xgb_t0_thr, xgb_t0_yes, xgb_t0_no, xgb_t0_off[t], x);
  else
    for (int t = 0; t < XGB_T1_NTREES; t++)
      s += xgb_tree(xgb_t1_feat, xgb_t1_thr, xgb_t1_yes, xgb_t1_no, xgb_t1_off[t], x);
  return s;
}

/* ---------------- ANN: INT8, float accumulation ---------------- */
static float h1[ANN_L0_OUT], h2[ANN_L1_OUT];

static void dense(const int8_t *w, const float *b, float s,
                  int in, int out, const float *src, float *dst, bool relu) {
  for (int o = 0; o < out; o++) {
    const int8_t *wr = w + (size_t)o * in;
    float acc = 0.0f;
    for (int i = 0; i < in; i++) acc += src[i] * ((float)wr[i] * s);
    acc += b[o];
    dst[o] = (relu && acc < 0.0f) ? 0.0f : acc;
  }
}
static void ann_predict(const float *x, float *out) {
  dense(ann_w0, ann_b0, ann_s0, ANN_L0_IN, ANN_L0_OUT, x,  h1, true);
  dense(ann_w1, ann_b1, ann_s1, ANN_L1_IN, ANN_L1_OUT, h1, h2, true);
  dense(ann_w2, ann_b2, ann_s2, ANN_L2_IN, ANN_L2_OUT, h2, out, false);
}

/* ---------------- auxilliary ---------------- */
static size_t xgb_flash_bytes() {
  return sizeof(xgb_t0_feat) + sizeof(xgb_t0_thr) + sizeof(xgb_t0_yes) +
         sizeof(xgb_t0_no)   + sizeof(xgb_t0_off) +
         sizeof(xgb_t1_feat) + sizeof(xgb_t1_thr) + sizeof(xgb_t1_yes) +
         sizeof(xgb_t1_no)   + sizeof(xgb_t1_off);
}
static size_t ann_flash_bytes() {
  return sizeof(ann_w0) + sizeof(ann_b0) + sizeof(ann_w1) + sizeof(ann_b1) +
         sizeof(ann_w2) + sizeof(ann_b2);
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println();
  Serial.println(F("==== ESP32 EDGE BENCHMARK ===="));

  Serial.print(F("Chip model     : ")); Serial.println(ESP.getChipModel());
  Serial.print(F("Chip revision  : ")); Serial.println(ESP.getChipRevision());
  Serial.print(F("Cores          : ")); Serial.println(ESP.getChipCores());
  Serial.print(F("CPU frequency  : ")); Serial.print(getCpuFrequencyMhz()); Serial.println(F(" MHz"));
  Serial.print(F("Flash size     : ")); Serial.println(ESP.getFlashChipSize());
  Serial.print(F("Free heap      : ")); Serial.println(ESP.getFreeHeap());
  Serial.print(F("Sketch size    : ")); Serial.println(ESP.getSketchSize());
  Serial.println();

  /* --- 1. check --- */
  float o[2];
  float x0 = xgb_predict(0, TV_XGB), x1 = xgb_predict(1, TV_XGB);
  ann_predict(TV_ANN, o);

  Serial.println(F("--- correctness (has to be equal to reference) ---"));
  Serial.print(F("XGB  on-device : ")); Serial.print(x0,8); Serial.print("  "); Serial.println(x1,8);
  Serial.print(F("XGB  reference : ")); Serial.print(REF_XGB0,8); Serial.print("  "); Serial.println(REF_XGB1,8);
  Serial.print(F("ANN  on-device : ")); Serial.print(o[0],8); Serial.print("  "); Serial.println(o[1],8);
  Serial.print(F("ANN  reference : ")); Serial.print(REF_ANN0,8); Serial.print("  "); Serial.println(REF_ANN1,8);

  bool ok = fabsf(x0 - REF_XGB0) < 1e-4f && fabsf(x1 - REF_XGB1) < 1e-4f &&
            fabsf(o[0] - REF_ANN0) < 1e-4f && fabsf(o[1] - REF_ANN1) < 1e-4f;
  Serial.print(F("VERDICT        : ")); Serial.println(ok ? F("PASS") : F("FAIL - ne meri dalje, javi mi"));
  Serial.println();

  /* --- 2. memory --- */
  Serial.println(F("--- memory ---"));
  Serial.print(F("XGBoost flash  : ")); Serial.print((unsigned long)xgb_flash_bytes());
  Serial.print(F(" B | nodes ")); Serial.print((int)(sizeof(xgb_t0_feat)/sizeof(int16_t)+sizeof(xgb_t1_feat)/sizeof(int16_t)));
  Serial.print(F(" | trees ")); Serial.print(XGB_T0_NTREES); Serial.print(F("+")); Serial.println(XGB_T1_NTREES);
  Serial.print(F("XGBoost RAM    : ")); Serial.print((unsigned long)(NFEAT*sizeof(float))); Serial.println(F(" B (input vector)"));
  Serial.print(F("ANN flash      : ")); Serial.print((unsigned long)ann_flash_bytes()); Serial.println(F(" B (INT8  + FP32 bias)"));
  Serial.print(F("ANN RAM        : ")); Serial.print((unsigned long)((ANN_IN+ANN_L0_OUT+ANN_L1_OUT)*sizeof(float))); Serial.println(F(" B"));
  Serial.println();

  /* --- 3. latency --- */
  Serial.println(F("--- latency ---"));
  volatile float sink = 0;
  uint32_t t0, dt;
  uint32_t best, worst; uint64_t sum;


  static float xin[NFEAT];
  static float ain[ANN_IN];

  best = 0xFFFFFFFF; worst = 0; sum = 0;
  for (int r = 0; r < REPS; r++) {
    for (int i = 0; i < NFEAT; i++) xin[i] = TV_XGB[i];
    xin[0] += (float)r * 1e-6f;                
    t0 = micros();
    sink += xgb_predict(0, xin) + xgb_predict(1, xin);
    dt = micros() - t0;
    sum += dt; if (dt < best) best = dt; if (dt > worst) worst = dt;
  }
  Serial.print(F("XGBoost (2 targets) : mean ")); Serial.print((float)sum/REPS, 3);
  Serial.print(F(" us | min ")); Serial.print((unsigned long)best);
  Serial.print(F(" | max ")); Serial.print((unsigned long)worst);
  Serial.print(F(" | n=")); Serial.println(REPS);

  best = 0xFFFFFFFF; worst = 0; sum = 0;
  for (int r = 0; r < REPS; r++) {
    for (int i = 0; i < ANN_IN; i++) ain[i] = TV_ANN[i];
    ain[0] += (float)r * 1e-6f;
    t0 = micros();
    ann_predict(ain, o);
    dt = micros() - t0;
    sum += dt; if (dt < best) best = dt; if (dt > worst) worst = dt;
    sink += o[0];
  }
  Serial.print(F("ANN INT8 (2 targets): mean ")); Serial.print((float)sum/REPS, 3);
  Serial.print(F(" us | min ")); Serial.print((unsigned long)best);
  Serial.print(F(" | max ")); Serial.print((unsigned long)worst);
  Serial.print(F(" | n=")); Serial.println(REPS);

  // Control: an empty loop with the same input copying, used to subtract measurement overhead.
  best = 0xFFFFFFFF; worst = 0; sum = 0;
  for (int r = 0; r < REPS; r++) {
    for (int i = 0; i < NFEAT; i++) xin[i] = TV_XGB[i];
    xin[0] += (float)r * 1e-6f;
    t0 = micros();
    sink += xin[0];
    dt = micros() - t0;
    sum += dt; if (dt < best) best = dt; if (dt > worst) worst = dt;
  }
  Serial.print(F("Kontrola (rezija)   : mean ")); Serial.print((float)sum/REPS, 3);
  Serial.print(F(" us | min ")); Serial.print((unsigned long)best);
  Serial.print(F(" | max ")); Serial.println((unsigned long)worst);

  Serial.print(F("\n(sink=")); Serial.print((float)sink+o[0],4);
  Serial.println(F("==== END ===="));
}

void loop() { delay(10000); }
