//+------------------------------------------------------------------+
//| OpenCL STFT Spectral Stabilizer (new file, original untouched)   |
//| 12 cycles + dominant mag/phase/freq via large-window FFT         |
//+------------------------------------------------------------------+
#property strict
#property indicator_separate_window
#property indicator_buffers 16
#property indicator_plots   13

#property indicator_label1  "P1"
#property indicator_type1   DRAW_LINE
#property indicator_label2  "P2"
#property indicator_type2   DRAW_LINE
#property indicator_label3  "P3"
#property indicator_type3   DRAW_LINE
#property indicator_label4  "P4"
#property indicator_type4   DRAW_LINE
#property indicator_label5  "P5"
#property indicator_type5   DRAW_LINE
#property indicator_label6  "P6"
#property indicator_type6   DRAW_LINE
#property indicator_label7  "P7"
#property indicator_type7   DRAW_LINE
#property indicator_label8  "P8"
#property indicator_type8   DRAW_LINE
#property indicator_label9  "P9"
#property indicator_type9   DRAW_LINE
#property indicator_label10 "P10"
#property indicator_type10  DRAW_LINE
#property indicator_label11 "P11"
#property indicator_type11  DRAW_LINE
#property indicator_label12 "P12"
#property indicator_type12  DRAW_LINE
#property indicator_label13 "WAVE"
#property indicator_type13  DRAW_LINE

// --- VkFFT DLL (OpenCL default; switch to CUDA by changing include)
#include <vkfft_exports_import_opencl.mqh>
// #include <vkfft_exports_import_cuda.mqh>

// --- Inputs
enum WINDOW_TYPE
  {
   WINDOW_NONE   = 0,
   WINDOW_HANN   = 1,
   WINDOW_HAMMING= 2,
   WINDOW_BLACKMAN=3
  };

input int         InpSTFTWindow      = 65536;   // FFT window (power of 2, max 131072)
input int         InpGpuMaxWindow    = 32768;   // Max FFT window for GPU (0=use full)
input int         InpMinPeriod       = 18;      // Min period (bars)
input int         InpMaxPeriod       = 52;      // Max period (bars)
input int         InpTimerMs         = 10;      // Timer interval (ms)
input WINDOW_TYPE InpWindowType      = WINDOW_HANN;
input double      InpSpecSmoothAlpha = 0.20;    // 0..1 smoothing for spectrum
input int         InpOpenCLDevice    = 0;       // OpenCL device ordinal (DLL uses first GPU)
input bool        InpUseHilbertDominant = true; // Instantaneous phase/freq for dominant
input bool        InpDebugOpenCL     = true;
input bool        InpUseVkfftAsync  = true;    // Use DLL begin/finish (no indicator-side async)
input bool        InpShowOnlySelected = true;   // Show only selected buffer
input bool        InpButtonInMainWindow = false;// Button in main chart window
input int         InpButtonX         = 6;       // Button X offset
input int         InpButtonY         = 8;       // Button Y offset
input bool        InpShowStatus      = true;    // Show status label
input bool        InpStatusInMainWindow = false;// Status in main chart window
input int         InpStatusX         = 6;       // Status X offset
input int         InpStatusY         = 28;      // Status Y offset
input bool        InpStatusAutoStack = true;    // Auto place status below button when same window
input int         InpMinComputeMs    = 0;       // Min interval between computes (0=disabled)
input int         InpHistoryRetryMs  = 2000;    // Min interval between history requests
enum WAVE_MODE
  {
   WAVE_OFF      = 0,
   WAVE_DOMINANT = 1,
   WAVE_TOPK     = 2,
   WAVE_ALL      = 3
  };
input WAVE_MODE   InpWaveMode        = WAVE_TOPK; // Which cycles compose WAVE
input int         InpWaveTopK        = 6;         // Used for WAVE_TOPK (1..12)
input double      InpWaveGain        = 1.0;       // Wave amplitude multiplier
enum BASELINE_MODE
  {
   BASELINE_ZERO = 0,
   BASELINE_MEAN = 1,
   BASELINE_CLOSE0 = 2
  };
input BASELINE_MODE InpWaveBaseline  = BASELINE_CLOSE0; // Baseline for WAVE

// --- Buffers (12 cycles)
double Cycle1[], Cycle2[], Cycle3[], Cycle4[];
double Cycle5[], Cycle6[], Cycle7[], Cycle8[];
double Cycle9[], Cycle10[], Cycle11[], Cycle12[];
double Wave[];
// --- Hidden buffers for dominant metrics (mag/freq/phase)
double DomMag[], DomFreq[], DomPhase[];

// --- OpenCL handles
int g_ctx = 0;
int g_prog = 0;
int g_k_stage = 0;
int g_k_bitrev = 0;
int g_bufA = 0;
int g_bufB = 0;

// --- FFT state
int    g_N = 0;
int    g_logN = 0;
double g_window[];      // window coeffs
double g_in2[];         // interleaved complex input (2*N)
double g_fft2[];        // interleaved complex output (2*N)
double g_dom2[];        // interleaved complex dominant spectrum (2*N)
double g_dom_time2[];   // interleaved complex time-domain dominant (2*N)
double g_mag[];         // magnitude spectrum (N/2)
double g_mag_smooth[];  // smoothed magnitude (N/2)

// --- VkFFT plan
int g_vkf_plan = -1;

// --- Working
int   g_top_idx[12];
double g_top_mag[12];
double g_top_phase[12];

double g_dom_phase_inst = EMPTY_VALUE;
double g_dom_freq_inst  = EMPTY_VALUE;
bool   g_dom_hilbert_ok = false;

string g_plot_names[13] = { "P1","P2","P3","P4","P5","P6","P7","P8","P9","P10","P11","P12","WAVE" };
string g_btn_name = "HZP_BufferToggle";
int    g_visible_plot = 12; // default to WAVE
string g_indicator_shortname = "OpenCL_STFT";
string g_status_name = "HZP_Status";
string g_status_name2 = "HZP_Status2";
int    g_requested_N = 0;
string g_status_text = "";
int    g_status_progress = -1;
uint   g_last_status_tick = 0;
string g_spinner = "|/-\\";
int    g_spinner_idx = 0;
bool   g_is_busy = false;
uint   g_last_compute_tick = 0;
uint   g_last_hist_tick = 0;

int g_last_bars = 0;
bool g_ready = false;
bool g_fft_pending = false;
bool g_fft_ready = false;
int g_latest_t = -1;
double g_last_baseline = 0.0;
double g_last_close0 = 0.0;
double g_last_close1 = 0.0;
double g_last_price_std = 0.0;

//+------------------------------------------------------------------+
int Pow2Clamp(int n)
{
   if(n < 1024) n = 1024;
   if(n > 131072) n = 131072;
   // clamp to nearest lower power of 2
   int p = 1;
   while((p << 1) <= n) p <<= 1;
   return p;
}

int ILog2(int n)
{
   int r = 0;
   while((1 << r) < n) r++;
   return r;
}

bool EnsureHistory(const int desired)
{
   if(desired <= 0) return false;
   MqlRates rates[];
   int got = CopyRates(_Symbol, _Period, 0, desired, rates);
   return (got > 0);
}

int GetIndicatorSubwindow()
{
   if(InpButtonInMainWindow)
      return 0;
   int subwin = ChartWindowFind(0, g_indicator_shortname);
   if(subwin < 0) subwin = 0;
   return subwin;
}

int GetStatusSubwindow()
{
   if(InpStatusInMainWindow)
      return 0;
   int subwin = ChartWindowFind(0, g_indicator_shortname);
   if(subwin < 0) subwin = 0;
   return subwin;
}

void UpdateToggleLabel()
{
   string label = ">> " + g_plot_names[g_visible_plot];
   ObjectSetString(0, g_btn_name, OBJPROP_TEXT, label);
}

void ApplyVisiblePlot()
{
   int total = 13;
   for(int i=0;i<total;i++)
     {
      int draw = DRAW_LINE;
      if(InpShowOnlySelected && i != g_visible_plot)
         draw = DRAW_NONE;
      PlotIndexSetInteger(i, PLOT_DRAW_TYPE, draw);
     }
}

void CreateToggleButton()
{
   int subwin = GetIndicatorSubwindow();
   if(ObjectFind(0, g_btn_name) >= 0)
      ObjectDelete(0, g_btn_name);
   if(!ObjectCreate(0, g_btn_name, OBJ_BUTTON, subwin, 0, 0))
      return;
   ObjectSetInteger(0, g_btn_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, g_btn_name, OBJPROP_XDISTANCE, InpButtonX);
   ObjectSetInteger(0, g_btn_name, OBJPROP_YDISTANCE, InpButtonY);
   ObjectSetInteger(0, g_btn_name, OBJPROP_XSIZE, 64);
   ObjectSetInteger(0, g_btn_name, OBJPROP_YSIZE, 18);
   ObjectSetInteger(0, g_btn_name, OBJPROP_FONTSIZE, 8);
   ObjectSetInteger(0, g_btn_name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, g_btn_name, OBJPROP_BACK, false);
   UpdateToggleLabel();
}

void RemoveToggleButton()
{
   if(ObjectFind(0, g_btn_name) >= 0)
      ObjectDelete(0, g_btn_name);
}

void CreateStatusLabel()
{
   if(!InpShowStatus) return;
   int subwin = GetStatusSubwindow();
   if(ObjectFind(0, g_status_name) >= 0)
      ObjectDelete(0, g_status_name);
   if(ObjectFind(0, g_status_name2) >= 0)
      ObjectDelete(0, g_status_name2);

   int y = InpStatusY;
   if(InpStatusAutoStack && subwin == GetIndicatorSubwindow())
      y = InpButtonY + 24;

   if(ObjectCreate(0, g_status_name, OBJ_LABEL, subwin, 0, 0))
     {
      ObjectSetInteger(0, g_status_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, g_status_name, OBJPROP_XDISTANCE, InpStatusX);
      ObjectSetInteger(0, g_status_name, OBJPROP_YDISTANCE, y);
      ObjectSetInteger(0, g_status_name, OBJPROP_FONTSIZE, 8);
      ObjectSetInteger(0, g_status_name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, g_status_name, OBJPROP_BACK, false);
      ObjectSetInteger(0, g_status_name, OBJPROP_COLOR, clrSilver);
      ObjectSetString(0, g_status_name, OBJPROP_TEXT, "Status: init");
     }

   if(ObjectCreate(0, g_status_name2, OBJ_LABEL, subwin, 0, 0))
     {
      ObjectSetInteger(0, g_status_name2, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, g_status_name2, OBJPROP_XDISTANCE, InpStatusX);
      ObjectSetInteger(0, g_status_name2, OBJPROP_YDISTANCE, y + 14);
      ObjectSetInteger(0, g_status_name2, OBJPROP_FONTSIZE, 8);
      ObjectSetInteger(0, g_status_name2, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, g_status_name2, OBJPROP_BACK, false);
      ObjectSetInteger(0, g_status_name2, OBJPROP_COLOR, clrSilver);
      ObjectSetString(0, g_status_name2, OBJPROP_TEXT, "Progresso: 0%");
     }
}

void RemoveStatusLabel()
{
   if(ObjectFind(0, g_status_name) >= 0)
      ObjectDelete(0, g_status_name);
   if(ObjectFind(0, g_status_name2) >= 0)
      ObjectDelete(0, g_status_name2);
}

void UpdateStatus(const string text, int progress)
{
   if(!InpShowStatus) return;
   bool show_progress = (progress >= 0);
   if(show_progress)
     {
      if(progress > 100) progress = 100;
      if(progress < 0) progress = 0;
     }
   uint now = (uint)GetTickCount();
   if(text == g_status_text && progress == g_status_progress && (now - g_last_status_tick) < 200)
      return;
   g_status_text = text;
   g_status_progress = progress;
   g_last_status_tick = now;
   bool busy = (text != "idle" && text != "ready" && StringFind(text, "waiting") < 0);
   if(busy)
     {
      g_spinner_idx = (g_spinner_idx + 1) % StringLen(g_spinner);
     }
   string spin = busy ? StringSubstr(g_spinner, g_spinner_idx, 1) : " ";
   string line = "Status: " + text + " " + spin;
   string line2 = show_progress ? ("Progresso: " + IntegerToString(progress) + "%") : "Progresso: --";
   ObjectSetString(0, g_status_name, OBJPROP_TEXT, line);
   ObjectSetString(0, g_status_name2, OBJPROP_TEXT, line2);
   color c = clrSilver;
   if(StringFind(text, "loading") >= 0) c = clrGold;
   else if(busy) c = clrDodgerBlue;
   ObjectSetInteger(0, g_status_name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, g_status_name2, OBJPROP_COLOR, c);
}

void NextVisiblePlot()
{
   for(int i=0;i<13;i++)
     {
      g_visible_plot = (g_visible_plot + 1) % 13;
      if(InpWaveMode == WAVE_OFF && g_visible_plot == 12)
         continue;
      break;
     }
   ApplyVisiblePlot();
   UpdateToggleLabel();
   ChartRedraw();
}

void BuildWindow()
{
   ArrayResize(g_window, g_N);
   if(InpWindowType == WINDOW_NONE)
     {
      for(int i=0;i<g_N;i++) g_window[i] = 1.0;
      return;
     }
   double denom = (g_N > 1) ? (double)(g_N - 1) : 1.0;
   for(int i=0;i<g_N;i++)
     {
      double x = 2.0 * M_PI * (double)i / denom;
      double w = 1.0;
      if(InpWindowType == WINDOW_HANN)
         w = 0.5 - 0.5 * MathCos(x);
      else if(InpWindowType == WINDOW_HAMMING)
         w = 0.54 - 0.46 * MathCos(x);
      else if(InpWindowType == WINDOW_BLACKMAN)
         w = 0.42 - 0.5 * MathCos(x) + 0.08 * MathCos(2.0*x);
      g_window[i] = w;
     }
}

string BuildKernelSource()
{
   string s;
   s += "#define M_PI_F 3.14159265358979323846f\n";
   s += "__kernel void bit_reverse(__global const float2* in, __global float2* out, int logN)\n";
   s += "{\n";
   s += "  int gid = get_global_id(0);\n";
   s += "  uint x = (uint)gid;\n";
   s += "  uint rev = 0;\n";
   s += "  for(int i=0;i<logN;i++){ rev = (rev << 1) | (x & 1); x >>= 1; }\n";
   s += "  out[rev] = in[gid];\n";
   s += "}\n";

   s += "__kernel void fft_stage(__global const float2* in, __global float2* out, int stage, int inverse)\n";
   s += "{\n";
   s += "  int gid = get_global_id(0);\n";
   s += "  int m = 1 << (stage + 1);\n";
   s += "  int m2 = m >> 1;\n";
   s += "  int j = gid & (m2 - 1);\n";
   s += "  int block = gid >> stage;\n";
   s += "  int i1 = block * m + j;\n";
   s += "  int i2 = i1 + m2;\n";
   s += "  float angle = (inverse != 0 ? 2.0f : -2.0f) * M_PI_F * (float)j / (float)m;\n";
   s += "  float c = native_cos(angle);\n";
   s += "  float s = native_sin(angle);\n";
   s += "  float2 u = in[i1];\n";
   s += "  float2 v = in[i2];\n";
   s += "  float2 t = (float2)(v.x*c - v.y*s, v.x*s + v.y*c);\n";
   s += "  out[i1] = (float2)(u.x + t.x, u.y + t.y);\n";
   s += "  out[i2] = (float2)(u.x - t.x, u.y - t.y);\n";
   s += "}\n";
   return s;
}

bool InitOpenCL()
{
   g_ctx = CLContextCreate(InpOpenCLDevice);
   if(g_ctx <= 0)
     {
      Print("OpenCL: CLContextCreate falhou err=", GetLastError());
      return false;
     }

   string src = BuildKernelSource();
   string buildlog = "";
   g_prog = CLProgramCreate(g_ctx, src, buildlog);
   if(g_prog <= 0)
     {
      Print("OpenCL: CLProgramCreate falhou err=", GetLastError());
      if(buildlog != "") Print(buildlog);
      return false;
     }

   g_k_bitrev = CLKernelCreate(g_prog, "bit_reverse");
   g_k_stage  = CLKernelCreate(g_prog, "fft_stage");
   if(g_k_bitrev <= 0 || g_k_stage <= 0)
     {
      Print("OpenCL: CLKernelCreate falhou err=", GetLastError());
      return false;
     }

   uint bytes = (uint)(2 * g_N * sizeof(double));
   g_bufA = CLBufferCreate(g_ctx, bytes, CL_MEM_READ_WRITE);
   g_bufB = CLBufferCreate(g_ctx, bytes, CL_MEM_READ_WRITE);
   if(g_bufA <= 0 || g_bufB <= 0)
     {
      Print("OpenCL: CLBufferCreate falhou err=", GetLastError());
      return false;
     }

   if(InpDebugOpenCL)
      Print("OpenCL init OK. N=", g_N, " logN=", g_logN);

   return true;
}

void FreeOpenCL()
{
   if(g_bufA>0) CLBufferFree(g_bufA);
   if(g_bufB>0) CLBufferFree(g_bufB);
   if(g_k_stage>0) CLKernelFree(g_k_stage);
   if(g_k_bitrev>0) CLKernelFree(g_k_bitrev);
   if(g_prog>0) CLProgramFree(g_prog);
   if(g_ctx>0) CLContextFree(g_ctx);
   g_bufA = g_bufB = g_k_stage = g_k_bitrev = g_prog = g_ctx = 0;
}

string VkfLastError(int h)
{
   uchar buf[256];
   int n = vkf_last_error_utf8(h, buf, 255);
   if(n <= 0) return "";
   return CharArrayToString(buf, 0, n);
}

bool InitVkFFT()
{
   int sizes[1]; sizes[0] = g_N;
   int zL[1]; zL[0] = 0;
   int zR[1]; zR[0] = 0;
   g_vkf_plan = vkf_plan_create(1, sizes, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, zL, zR);
   if(g_vkf_plan < 0)
     {
      Print("VkFFT: plan_create falhou: ", VkfLastError(-1));
      return false;
     }
   vkf_set_debug(InpDebugOpenCL ? 1 : 0);
   if(InpDebugOpenCL)
      Print("VkFFT init OK. N=", g_N);
   return true;
}

void FreeVkFFT()
{
   if(g_vkf_plan >= 0)
     {
      vkf_plan_destroy(g_vkf_plan);
      g_vkf_plan = -1;
     }
}

bool RunFFT(bool inverse, double &in2[], double &out2[])
{
   if(g_vkf_plan < 0) return false;
   if(inverse)
     {
      int rc = vkf_exec_inverse(g_vkf_plan, in2, out2);
      return (rc == 0);
     }
   if(InpUseVkfftAsync)
     {
      // InpUseVkfftAsync is handled in OnTimer (submit/poll/fetch) for forward
      return false;
     }
   int rc = inverse ? vkf_exec_inverse(g_vkf_plan, in2, out2)
                    : vkf_exec_forward(g_vkf_plan, in2, out2);
   return (rc == 0);
}

void SelectTopCycles(int min_index, int max_index)
{
   for(int i=0;i<12;i++){ g_top_idx[i] = -1; g_top_mag[i] = 0.0; g_top_phase[i] = 0.0; }

   for(int i=min_index; i<=max_index; i++)
     {
      double p = g_mag_smooth[i];
      if(p <= 0.0) continue;
      // insert into top list
      for(int k=0;k<12;k++)
        {
         if(p > g_top_mag[k])
           {
            for(int j=11;j>k;j--){ g_top_mag[j]=g_top_mag[j-1]; g_top_idx[j]=g_top_idx[j-1]; g_top_phase[j]=g_top_phase[j-1]; }
            g_top_mag[k] = p;
            g_top_idx[k] = i;
            g_top_phase[k] = MathArctan2((double)g_fft2[2*i+1], (double)g_fft2[2*i]);
            break;
           }
        }
     }
}

void UpdateBuffers(const int bars)
{
   int maxFill = (bars < g_N) ? bars : g_N;
   if(maxFill <= 0) return;
   if(g_latest_t < 0 || g_latest_t >= g_N) return;

   double twoPi = 2.0 * M_PI;
   bool wave_enabled = (InpWaveMode != WAVE_OFF);
   int topk = InpWaveTopK;
   if(topk < 1) topk = 1;
   if(topk > 12) topk = 12;
   bool wave_used = false;

   Cycle1[0]=Cycle2[0]=Cycle3[0]=Cycle4[0]=EMPTY_VALUE;
   Cycle5[0]=Cycle6[0]=Cycle7[0]=Cycle8[0]=EMPTY_VALUE;
   Cycle9[0]=Cycle10[0]=Cycle11[0]=Cycle12[0]=EMPTY_VALUE;
   Wave[0] = wave_enabled ? 0.0 : EMPTY_VALUE;

   int t = g_latest_t; // latest sample index in FFT window
   int t_prev = (t > 0) ? (t - 1) : t;
   double wave_sum = 0.0;
   double wave_sum_prev = 0.0;
   double wave_amp2 = 0.0;

   // Fill cycles (bar 0 only)
   for(int c=0;c<12;c++)
     {
      int idx = g_top_idx[c];
      if(idx <= 0) continue;
      double mag = g_top_mag[c];
      double amp = (idx == 0 || idx == g_N/2) ? (mag / g_N) : (2.0 * mag / g_N);
      double phase = g_top_phase[c];
      double freq = (double)idx / (double)g_N; // cycles per bar
      bool use_in_wave = false;
      if(wave_enabled)
        {
         if(InpWaveMode == WAVE_ALL)
            use_in_wave = true;
         else if(InpWaveMode == WAVE_DOMINANT && c == 0)
            use_in_wave = true;
         else if(InpWaveMode == WAVE_TOPK && c < topk)
            use_in_wave = true;
        }

      double value = amp * MathCos(twoPi * freq * (double)t + phase);
      double value_prev = amp * MathCos(twoPi * freq * (double)t_prev + phase);
      switch(c)
        {
         case 0: Cycle1[0] = value; break;
         case 1: Cycle2[0] = value; break;
         case 2: Cycle3[0] = value; break;
         case 3: Cycle4[0] = value; break;
         case 4: Cycle5[0] = value; break;
         case 5: Cycle6[0] = value; break;
         case 6: Cycle7[0] = value; break;
         case 7: Cycle8[0] = value; break;
         case 8: Cycle9[0] = value; break;
         case 9: Cycle10[0] = value; break;
         case 10: Cycle11[0] = value; break;
         case 11: Cycle12[0] = value; break;
        }
      if(use_in_wave)
        {
         wave_sum += value;
         wave_sum_prev += value_prev;
         wave_amp2 += (amp * amp);
         wave_used = true;
        }
     }

   if(wave_enabled)
     {
      if(!wave_used)
         Wave[0] = EMPTY_VALUE;
      else if(InpWaveGain != 1.0)
         wave_sum *= InpWaveGain;
      // Scale wave amplitude to price std-dev
      double wave_rms = (wave_amp2 > 0.0) ? MathSqrt(0.5 * wave_amp2) : 0.0;
      if(wave_rms > 0.0 && g_last_price_std > 0.0)
        {
         double scale = g_last_price_std / wave_rms;
         wave_sum *= scale;
         wave_sum_prev *= scale;
        }
      // Align wave direction with price slope
      double price_delta = g_last_close0 - g_last_close1;
      double wave_delta = wave_sum - wave_sum_prev;
      if(price_delta * wave_delta < 0.0)
         wave_sum = -wave_sum;
      Wave[0] = wave_sum;
      if(Wave[0] != EMPTY_VALUE)
         Wave[0] += g_last_baseline;
     }

   // Dominant metrics (buffer[0] = latest)
   int dom = g_top_idx[0];
   if(dom > 0)
     {
      DomMag[0] = g_top_mag[0];
      if(g_dom_hilbert_ok)
        {
         DomFreq[0] = g_dom_freq_inst;
         DomPhase[0] = g_dom_phase_inst;
        }
      else
        {
         DomFreq[0] = (double)dom / (double)g_N; // cycles/bar
         DomPhase[0] = g_top_phase[0];
        }
     }
   else
     {
      DomMag[0] = EMPTY_VALUE;
      DomFreq[0] = EMPTY_VALUE;
      DomPhase[0] = EMPTY_VALUE;
     }
}

bool ComputeDominantHilbert(int dom_idx, double &phase_out, double &freq_out)
{
   if(dom_idx <= 0 || dom_idx >= g_N/2)
      return false;

   ArrayResize(g_dom2, 2*g_N);
   ArrayInitialize(g_dom2, 0.0);

   // Keep only dominant positive bin, scale by 2 for analytic signal
   g_dom2[2*dom_idx]   = 2.0 * g_fft2[2*dom_idx];
   g_dom2[2*dom_idx+1] = 2.0 * g_fft2[2*dom_idx+1];

   ArrayResize(g_dom_time2, 2*g_N);
   if(!RunFFT(true, g_dom2, g_dom_time2)) return false; // inverse FFT

   // scale by 1/N
   double invN = 1.0 / (double)g_N;
   g_dom_time2[2*(g_N-1)]   *= invN;
   g_dom_time2[2*(g_N-1)+1] *= invN;
   g_dom_time2[2*(g_N-2)]   *= invN;
   g_dom_time2[2*(g_N-2)+1] *= invN;

   double re1 = g_dom_time2[2*(g_N-1)];
   double im1 = g_dom_time2[2*(g_N-1)+1];
   double re0 = g_dom_time2[2*(g_N-2)];
   double im0 = g_dom_time2[2*(g_N-2)+1];

   double p1 = MathArctan2(im1, re1);
   double p0 = MathArctan2(im0, re0);
   double dphi = p1 - p0;
   if(dphi > M_PI) dphi -= 2.0 * M_PI;
   if(dphi < -M_PI) dphi += 2.0 * M_PI;

   phase_out = p1;
   freq_out  = dphi / (2.0 * M_PI); // cycles/bar
   if(freq_out < 0) freq_out = -freq_out;

   return true;
}

void ClearBuffers(const int bars)
{
   for(int i=0;i<bars;i++)
     {
      Cycle1[i]=Cycle2[i]=Cycle3[i]=Cycle4[i]=EMPTY_VALUE;
      Cycle5[i]=Cycle6[i]=Cycle7[i]=Cycle8[i]=EMPTY_VALUE;
      Cycle9[i]=Cycle10[i]=Cycle11[i]=Cycle12[i]=EMPTY_VALUE;
      Wave[i]=EMPTY_VALUE;
     }
   DomMag[0]=DomFreq[0]=DomPhase[0]=EMPTY_VALUE;
   g_dom_hilbert_ok = false;
   g_dom_phase_inst = EMPTY_VALUE;
   g_dom_freq_inst = EMPTY_VALUE;
}

//+------------------------------------------------------------------+
int OnInit()
{
   IndicatorSetString(INDICATOR_SHORTNAME, g_indicator_shortname);
   UpdateStatus("initializing", -1);
   g_requested_N = Pow2Clamp(InpSTFTWindow);
   g_N = g_requested_N;
   if(InpGpuMaxWindow > 0)
     {
      int cap = Pow2Clamp(InpGpuMaxWindow);
      if(cap > 0 && g_N > cap)
        g_N = cap;
     }
   UpdateStatus("loading history", -1);
   EnsureHistory(g_N);
   g_logN = ILog2(g_N);

   BuildWindow();

   ArrayResize(g_in2, 2*g_N);
   ArrayResize(g_fft2, 2*g_N);
   ArrayResize(g_mag, g_N/2);
   ArrayResize(g_mag_smooth, g_N/2);
   ArrayInitialize(g_mag_smooth, 0.0);

   // Buffers
   SetIndexBuffer(0, Cycle1, INDICATOR_DATA);
   SetIndexBuffer(1, Cycle2, INDICATOR_DATA);
   SetIndexBuffer(2, Cycle3, INDICATOR_DATA);
   SetIndexBuffer(3, Cycle4, INDICATOR_DATA);
   SetIndexBuffer(4, Cycle5, INDICATOR_DATA);
   SetIndexBuffer(5, Cycle6, INDICATOR_DATA);
   SetIndexBuffer(6, Cycle7, INDICATOR_DATA);
   SetIndexBuffer(7, Cycle8, INDICATOR_DATA);
   SetIndexBuffer(8, Cycle9, INDICATOR_DATA);
   SetIndexBuffer(9, Cycle10, INDICATOR_DATA);
   SetIndexBuffer(10, Cycle11, INDICATOR_DATA);
   SetIndexBuffer(11, Cycle12, INDICATOR_DATA);
   SetIndexBuffer(12, Wave, INDICATOR_DATA);
   SetIndexBuffer(13, DomMag, INDICATOR_DATA);
   SetIndexBuffer(14, DomFreq, INDICATOR_DATA);
   SetIndexBuffer(15, DomPhase, INDICATOR_DATA);

   ArraySetAsSeries(Cycle1, true); ArraySetAsSeries(Cycle2, true);
   ArraySetAsSeries(Cycle3, true); ArraySetAsSeries(Cycle4, true);
   ArraySetAsSeries(Cycle5, true); ArraySetAsSeries(Cycle6, true);
   ArraySetAsSeries(Cycle7, true); ArraySetAsSeries(Cycle8, true);
   ArraySetAsSeries(Cycle9, true); ArraySetAsSeries(Cycle10, true);
   ArraySetAsSeries(Cycle11, true); ArraySetAsSeries(Cycle12, true);
   ArraySetAsSeries(Wave, true);
   ArraySetAsSeries(DomMag, true); ArraySetAsSeries(DomFreq, true); ArraySetAsSeries(DomPhase, true);

   if(!InitVkFFT())
      return INIT_FAILED;

   if(InpWaveMode == WAVE_OFF)
      g_visible_plot = 0;
   ApplyVisiblePlot();
   CreateToggleButton();
   CreateStatusLabel();
   if(g_N < g_requested_N)
      UpdateStatus("ready (fft capped " + IntegerToString(g_N) + "/" + IntegerToString(g_requested_N) + ")", -1);
   else
      UpdateStatus("ready", -1);

   EventSetMillisecondTimer(InpTimerMs);
   g_ready = true;
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   RemoveToggleButton();
   RemoveStatusLabel();
   FreeVkFFT();
   g_ready = false;
}

void OnTimer()
{
   if(!g_ready) return;

   uint now = (uint)GetTickCount();
   if(g_is_busy) return;
   if(InpMinComputeMs > 0 && (now - g_last_compute_tick) < (uint)InpMinComputeMs)
      return;
   g_is_busy = true;

   // If async FFT is pending, only poll/fetch and then continue to spectrum
   if(InpUseVkfftAsync && g_fft_pending)
     {
      int st = vkf_async_poll(g_vkf_plan);
      if(st == 0)
        {
         UpdateStatus("running fft", -1);
         g_is_busy = false;
         return;
        }
      if(st < 0)
        {
         UpdateStatus("fft failed " + IntegerToString(st), -1);
         g_fft_pending = false;
         g_is_busy = false;
         return;
        }
      int frc = vkf_async_fetch(g_vkf_plan, g_fft2, 2*g_N);
      if(frc != 0)
        {
         UpdateStatus("fft fetch failed " + IntegerToString(frc), -1);
         g_fft_pending = false;
         g_is_busy = false;
         return;
        }
      g_fft_pending = false;
      g_fft_ready = true;
     }

   int bars = Bars(_Symbol, _Period);
   if(bars <= 0)
     {
      UpdateStatus("waiting data", -1);
      g_is_busy = false;
      return;
     }
   if(bars < g_N)
     {
      int pct = (int)MathMin(100.0, 100.0 * (double)bars / (double)g_N);
      UpdateStatus("waiting history " + IntegerToString(bars) + "/" + IntegerToString(g_N), pct);
      if(InpHistoryRetryMs > 0 && (now - g_last_hist_tick) >= (uint)InpHistoryRetryMs)
        {
         EnsureHistory(g_N);
         g_last_hist_tick = now;
        }
     }

   UpdateStatus("working", -1);
   double alpha = InpSpecSmoothAlpha;
   if(alpha < 0.0) alpha = 0.0; if(alpha > 1.0) alpha = 1.0;

   if(!g_fft_pending && !g_fft_ready)
     {
   // Load available prices (may be less than window; zero-pad both ends)
   double price[];
   int bars_avail = bars;
   if(bars_avail > g_N) bars_avail = g_N;
   ArrayResize(price, bars_avail);
   ArraySetAsSeries(price, true);
   int copied = CopyClose(_Symbol, _Period, 0, bars_avail, price);
   if(copied <= 0)
     {
      UpdateStatus("waiting data", -1);
      g_is_busy = false;
      return;
     }
   if(copied < bars_avail) bars_avail = copied;
   if(bars_avail <= 0)
     {
      UpdateStatus("waiting data", -1);
      g_is_busy = false;
      return;
     }
   g_last_close0 = price[0];
   g_last_close1 = (bars_avail > 1) ? price[1] : price[0];
   UpdateStatus("building window", -1);

   // Detrend by mean
   double mean = 0.0;
   for(int i=0;i<bars_avail;i++) mean += price[i];
   mean /= (double)bars_avail;
   double var = 0.0;
   for(int i=0;i<bars_avail;i++)
     {
      double d = price[i] - mean;
      var += d * d;
     }
   g_last_price_std = (bars_avail > 1) ? MathSqrt(var / (double)bars_avail) : 0.0;
   if(InpWaveBaseline == BASELINE_MEAN)
      g_last_baseline = mean;
   else if(InpWaveBaseline == BASELINE_CLOSE0)
      g_last_baseline = price[0];
   else
      g_last_baseline = 0.0;

   // Build complex input (oldest->newest) with symmetric zero padding
   int total_pad = g_N - bars_avail;
   if(total_pad < 0) total_pad = 0;
   if((total_pad % 2) != 0 && bars_avail > 1)
     {
      // enforce same number of zeros on both sides
      bars_avail -= 1;
      total_pad = g_N - bars_avail;
     }
   int pad = total_pad / 2;
   int pad_right = total_pad - pad;
   int data_start = pad;
   int data_end = pad + bars_avail; // exclusive
   g_latest_t = data_end - 1;

   for(int t=0; t<g_N; t++)
     {
      double x = 0.0;
      if(t >= data_start && t < data_end)
        {
         int data_idx = t - data_start;
         int bar = bars_avail - 1 - data_idx; // oldest at t=0
         x = price[bar] - mean;
        }
      double w = g_window[t];
      g_in2[2*t]   = x * w;
      g_in2[2*t+1] = 0.0;
     }

   UpdateStatus("running fft", -1);
   if(InpUseVkfftAsync)
     {
      int rc = vkf_async_submit_forward(g_vkf_plan, g_in2, 2*g_N);
      if(rc == 0)
        {
         g_fft_pending = true;
         g_is_busy = false;
         return;
        }
      UpdateStatus("fft submit failed " + IntegerToString(rc), -1);
      g_is_busy = false;
      return;
     }
   else
     {
      if(!RunFFT(false, g_in2, g_fft2))
        {
         UpdateStatus("fft failed", -1);
         g_is_busy = false;
         return;
        }
     }
   }

   if(InpUseVkfftAsync && g_fft_ready)
      g_fft_ready = false;

   // Magnitude
   UpdateStatus("spectrum", -1);
   int half = g_N / 2;
   ArrayResize(g_mag, half);
   for(int i=0;i<half;i++)
     {
      double re = g_fft2[2*i];
      double im = g_fft2[2*i+1];
      double m = MathSqrt(re*re + im*im);
      g_mag[i] = m;
      if(alpha == 0.0)
         g_mag_smooth[i] = g_mag[i];
      else
         g_mag_smooth[i] = (1.0 - alpha) * g_mag_smooth[i] + alpha * g_mag[i];
     }

   int min_index = (int)MathCeil((double)g_N / (double)InpMaxPeriod);
   int max_index = (int)MathFloor((double)g_N / (double)InpMinPeriod);
   if(min_index < 1) min_index = 1;
   if(max_index >= half) max_index = half - 1;
   if(min_index > max_index)
     {
      UpdateStatus("no spectrum range", -1);
      g_is_busy = false;
      return;
     }

   UpdateStatus("selecting cycles", -1);
   SelectTopCycles(min_index, max_index);

   // Dominant instantaneous phase/freq via Hilbert (optional)
   if(InpUseHilbertDominant && g_top_idx[0] > 0)
     {
      double ph=0.0, fr=0.0;
      g_dom_hilbert_ok = ComputeDominantHilbert(g_top_idx[0], ph, fr);
      if(g_dom_hilbert_ok) { g_dom_phase_inst = ph; g_dom_freq_inst = fr; }
      else { g_dom_phase_inst = EMPTY_VALUE; g_dom_freq_inst = EMPTY_VALUE; }
     }
   else
     {
      g_dom_hilbert_ok = false;
      g_dom_phase_inst = EMPTY_VALUE;
      g_dom_freq_inst = EMPTY_VALUE;
     }

   // update buffers
   UpdateStatus("updating buffers", -1);
   UpdateBuffers(bars);
   UpdateStatus("idle", -1);
   ChartRedraw();
   g_last_compute_tick = now;
   g_is_busy = false;
}

int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   if(prev_calculated == 0)
     {
      g_last_bars = rates_total;
      ClearBuffers(rates_total);
     }
   return rates_total;
}

void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
   if(id == CHARTEVENT_OBJECT_CLICK && sparam == g_btn_name)
      NextVisiblePlot();
}
