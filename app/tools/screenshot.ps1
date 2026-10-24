# Capture a screen, or a single window, to PNG so the assistant can check its
# own work.
#
# Written because the alternative was asking the user "what do you see?" after
# every UI change, which makes them QA for things that are trivially checkable.
# They should only have to do the two things software cannot: shoot the gun and
# move the camera.
#
#   -Index   which screen (0 = primary). Ignored when -Window is given.
#   -Window  capture the window whose title contains this, BY HANDLE, even if it
#            is buried behind other windows. This is the reliable mode:
#            SetForegroundWindow is routinely REFUSED when the caller is not
#            already the foreground process -- Windows just flashes the taskbar
#            button -- so "activate, then grab the screen" silently captures
#            whatever was on top instead. Observed exactly that.
#   -Out     output path
#   -Width   downscale to this width (0 = full size). A 4K grab is enormous and
#            nothing being checked here needs that much detail.
#   -List    print every visible top-level window title and exit.
param(
    [int]$Index = 0,
    [string]$Window = "",
    [string]$Out = "screen.png",
    [int]$Width = 1600,
    [switch]$List
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

Add-Type @"
using System;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;
public class Win {
    public delegate bool EnumProc(IntPtr h, IntPtr p);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr p);
    [DllImport("user32.dll")] public static extern int GetWindowTextLength(IntPtr h);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)]
      public static extern int GetWindowText(IntPtr h, StringBuilder s, int max);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr dc, uint flags);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }

    public static List<string> Titles = new List<string>();
    public static List<IntPtr> Handles = new List<IntPtr>();
    public static void Collect() {
        Titles.Clear(); Handles.Clear();
        EnumWindows(delegate(IntPtr h, IntPtr p) {
            if (!IsWindowVisible(h)) return true;
            int n = GetWindowTextLength(h);
            if (n == 0) return true;
            StringBuilder sb = new StringBuilder(n + 1);
            GetWindowText(h, sb, sb.Capacity);
            Titles.Add(sb.ToString()); Handles.Add(h);
            return true;
        }, IntPtr.Zero);
    }
}
"@

if ($List) {
    [Win]::Collect()
    [Win]::Titles | ForEach-Object { Write-Output $_ }
    exit 0
}

$bmp = $null
if ($Window -ne "") {
    [Win]::Collect()
    $hit = -1
    for ($i = 0; $i -lt [Win]::Titles.Count; $i++) {
        if ([Win]::Titles[$i] -like "*$Window*") { $hit = $i; break }
    }
    if ($hit -lt 0) {
        Write-Output "no window matching '$Window'. Visible windows:"
        [Win]::Titles | ForEach-Object { Write-Output "   $_" }
        exit 1
    }
    $hw = [Win]::Handles[$hit]
    $r = New-Object Win+RECT
    [void][Win]::GetWindowRect($hw, [ref]$r)
    $w = $r.Right - $r.Left
    $ht = $r.Bottom - $r.Top
    if ($w -le 0 -or $ht -le 0) { Write-Output "window has no size"; exit 1 }
    $bmp = New-Object System.Drawing.Bitmap $w, $ht
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $hdc = $g.GetHdc()
    # Flag 2 is PW_RENDERFULLCONTENT. Without it a hardware-composited window,
    # which is what Qt gives you, comes back blank.
    [void][Win]::PrintWindow($hw, $hdc, 2)
    $g.ReleaseHdc($hdc)
    $g.Dispose()
    Write-Output "captured window: $([Win]::Titles[$hit])  ($w x $ht)"
} else {
    $screens = [System.Windows.Forms.Screen]::AllScreens
    if ($Index -ge $screens.Count) {
        Write-Output "only $($screens.Count) screen(s); index $Index does not exist"
        exit 1
    }
    $b = $screens[$Index].Bounds
    $bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
    $g.Dispose()
    Write-Output "captured screen $Index  ($($b.Width) x $($b.Height))"
}

if ($Width -gt 0 -and $Width -lt $bmp.Width) {
    $h2 = [int]($bmp.Height * $Width / $bmp.Width)
    $small = New-Object System.Drawing.Bitmap $Width, $h2
    $sg = [System.Drawing.Graphics]::FromImage($small)
    $sg.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $sg.DrawImage($bmp, 0, 0, $Width, $h2)
    $sg.Dispose()
    $bmp.Dispose()
    $bmp = $small
}

$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()
Write-Output "saved $Out"
