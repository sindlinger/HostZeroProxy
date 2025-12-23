#property strict
#property description "EA: reads 12 cycle buffers from a custom indicator and colors labels by orientation"
#property version   "1.00"

input string          InpIndicatorName  = "TFT-HostComplete-Indicator_v4.0-Waveform-Top2---OK_VKFFT_NoDetrend";
input ENUM_TIMEFRAMES InpIndicatorTF    = PERIOD_CURRENT;
input int             InpBufferBase     = 0;
input int             InpCycleCount     = 12;

enum ORIENT_MODE
{
    ORIENT_SLOPE = 0,
    ORIENT_ZERO  = 1,
    ORIENT_NEIGHBOR_CROSS = 2
};

input ORIENT_MODE InpOrientation = ORIENT_SLOPE;
input double      InpEps         = 1e-6;
input int         InpUpdateMs    = 200; // 0 = only on tick

input color       InpUpColor     = clrLime;
input color       InpDownColor   = clrTomato;
input color       InpFlatColor   = clrSilver;

input int         InpCorner      = 0; // 0=left upper
input int         InpX           = 10;
input int         InpY           = 20;
input int         InpYStep       = 14;
input int         InpFontSize    = 9;

int    g_handle = INVALID_HANDLE;
string g_label_prefix = "HZP_Cycle_";

double g_cur_vals[];
double g_prev_vals[];

int SignValue(const double v, const double eps)
{
    if(MathAbs(v) <= eps)
        return 0;
    return (v > 0.0) ? 1 : -1;
}

string LabelName(const int idx)
{
    return g_label_prefix + IntegerToString(idx + 1);
}

void CreateLabel(const int idx)
{
    string name = LabelName(idx);
    if(ObjectFind(0, name) >= 0)
        return;
    if(!ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0))
    {
        PrintFormat("[EA] ObjectCreate failed for %s (err=%d)", name, GetLastError());
        return;
    }
    ObjectSetInteger(0, name, OBJPROP_CORNER, InpCorner);
    ObjectSetInteger(0, name, OBJPROP_XDISTANCE, InpX);
    ObjectSetInteger(0, name, OBJPROP_YDISTANCE, InpY + idx * InpYStep);
    ObjectSetInteger(0, name, OBJPROP_FONTSIZE, InpFontSize);
    ObjectSetString(0, name, OBJPROP_FONT, "Arial");
}

void UpdateLabel(const int idx, const string text, const color clr)
{
    string name = LabelName(idx);
    if(ObjectFind(0, name) < 0)
        CreateLabel(idx);
    ObjectSetString(0, name, OBJPROP_TEXT, text);
    ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
}

void ClearLabels()
{
    for(int i = 0; i < InpCycleCount; i++)
        ObjectDelete(0, LabelName(i));
}

bool ReadBuffers()
{
    int n = MathMax(1, InpCycleCount);
    ArrayResize(g_cur_vals, n);
    ArrayResize(g_prev_vals, n);

    for(int i = 0; i < n; i++)
    {
        double tmp[2];
        int copied = CopyBuffer(g_handle, InpBufferBase + i, 0, 2, tmp);
        if(copied != 2)
        {
            g_cur_vals[i] = EMPTY_VALUE;
            g_prev_vals[i] = EMPTY_VALUE;
            continue;
        }
        g_cur_vals[i] = tmp[0];
        g_prev_vals[i] = tmp[1];
    }
    return true;
}

void UpdateOrientation()
{
    if(g_handle == INVALID_HANDLE)
        return;

    ReadBuffers();

    int n = MathMax(1, InpCycleCount);
    for(int i = 0; i < n; i++)
    {
        double cur = g_cur_vals[i];
        double prev = g_prev_vals[i];
        if(cur == EMPTY_VALUE || prev == EMPTY_VALUE)
        {
            UpdateLabel(i, StringFormat("C%d: n/a", i + 1), clrGray);
            continue;
        }

        double diff = 0.0;
        if(InpOrientation == ORIENT_SLOPE)
        {
            diff = cur - prev;
        }
        else if(InpOrientation == ORIENT_ZERO)
        {
            diff = cur;
        }
        else // ORIENT_NEIGHBOR_CROSS
        {
            int j = (i < n - 1) ? (i + 1) : (i - 1);
            if(j < 0 || j >= n || g_cur_vals[j] == EMPTY_VALUE)
                diff = cur - prev;
            else
                diff = cur - g_cur_vals[j];
        }

        int s = SignValue(diff, InpEps);
        if(s > 0)
            UpdateLabel(i, StringFormat("C%d: UP", i + 1), InpUpColor);
        else if(s < 0)
            UpdateLabel(i, StringFormat("C%d: DOWN", i + 1), InpDownColor);
        else
            UpdateLabel(i, StringFormat("C%d: FLAT", i + 1), InpFlatColor);
    }
}

int OnInit()
{
    g_handle = iCustom(_Symbol, InpIndicatorTF, InpIndicatorName);
    if(g_handle == INVALID_HANDLE)
    {
        PrintFormat("[EA] iCustom failed (%s) err=%d", InpIndicatorName, GetLastError());
        return INIT_FAILED;
    }

    for(int i = 0; i < InpCycleCount; i++)
        CreateLabel(i);

    if(InpUpdateMs > 0)
        EventSetMillisecondTimer((uint)MathMax(1, InpUpdateMs));

    UpdateOrientation();
    return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
    if(InpUpdateMs > 0)
        EventKillTimer();
    ClearLabels();
    if(g_handle != INVALID_HANDLE)
    {
        IndicatorRelease(g_handle);
        g_handle = INVALID_HANDLE;
    }
}

void OnTick()
{
    if(InpUpdateMs <= 0)
        UpdateOrientation();
}

void OnTimer()
{
    UpdateOrientation();
}
