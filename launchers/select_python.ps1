# Chooses the python.exe that the launchers use and saves it in config\python_environment.json of the project folder.
# It is PowerShell (always on Windows) so that it works before there is any Python to run a window with.
#   -Root    the project folder (default: the folder above this one)
#   -Dialog  always ask; without it the saved Python is used while it is still there, and the window opens only if it is not
# It writes the chosen python.exe as the only output and exits with 0; it exits with 1 if nothing was chosen.
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot),
    [switch]$Dialog
)

$file = Join-Path $Root 'config\python_environment.json'

function Get-SavedPython {
    if (Test-Path -LiteralPath $file) {
        try { return [string](Get-Content -Raw -LiteralPath $file | ConvertFrom-Json).python } catch { return '' }
    }
    return ''
}

# '' if this Python can be used (3.10 or newer, with Tkinter that can open a window), else what is wrong
function Test-Python([string]$path) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return 'The file does not exist.' }
    try {
        $info = New-Object System.Diagnostics.ProcessStartInfo
        $info.FileName = $path
        $info.Arguments = '-c "import sys, tkinter; r = tkinter.Tk(); r.withdraw(); r.destroy(); print(sys.version_info[0], sys.version_info[1])"'
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $process = [System.Diagnostics.Process]::Start($info)
        $output = $process.StandardOutput.ReadToEnd()
        $errors = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
    } catch {
        return "This file cannot be run as Python: $($_.Exception.Message)"
    }
    if ($output.Trim() -notmatch '^\d+ \d+$') {
        if ($process.ExitCode -ne 0) { return "This Python cannot import Tkinter. $($errors.Trim())" }
        return 'This file does not behave like python.exe.'
    }
    if ($process.ExitCode -ne 0) { return "This Python cannot import Tkinter. $($errors.Trim())" }
    $version = $output.Trim().Split(' ')
    if ([int]$version[0] -lt 3 -or ([int]$version[0] -eq 3 -and [int]$version[1] -lt 10)) {
        return "This is Python $($version[0]).$($version[1]); Python 3.10 or newer is needed."
    }
    return ''
}

$saved = Get-SavedPython
if (-not $Dialog -and $saved -and (Test-Path -LiteralPath $saved)) {
    Write-Output $saved
    exit 0
}

# What the box shows at the start: the saved Python, else the first one in the PATH (not the Microsoft Store shortcut)
$initial = $saved
if (-not $initial) {
    $found = Get-Command python.exe -All -ErrorAction SilentlyContinue |
        Where-Object { $_.Source -notlike '*\WindowsApps\*' } | Select-Object -First 1
    if ($found) { $initial = $found.Source }
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()
function Color([string]$hex) { return [System.Drawing.ColorTranslator]::FromHtml($hex) }
$background = Color '#242424'; $fieldColor = Color '#595d61'; $foreground = Color '#dedede'
$muted = Color '#aaaeb2'; $accent = Color '#485968'; $border = Color '#444444'; $errorColor = Color '#e8785a'

function New-Label([string]$text, [int]$x, [int]$y, [int]$width, [int]$height, $color, $bold) {
    $label = New-Object System.Windows.Forms.Label
    $label.Text = $text
    $label.Location = New-Object System.Drawing.Point($x, $y)
    $label.Size = New-Object System.Drawing.Size($width, $height)
    $label.ForeColor = $color
    if ($bold) { $label.Font = New-Object System.Drawing.Font('Segoe UI', 11, [System.Drawing.FontStyle]::Bold) }
    return $label
}

function New-Button([string]$text, [int]$x, [int]$y, [int]$width) {
    $button = New-Object System.Windows.Forms.Button
    $button.Text = $text
    $button.Location = New-Object System.Drawing.Point($x, $y)
    $button.Size = New-Object System.Drawing.Size($width, 30)
    $button.FlatStyle = 'Flat'
    $button.BackColor = $fieldColor
    $button.ForeColor = $foreground
    $button.FlatAppearance.BorderColor = $border
    $button.FlatAppearance.MouseOverBackColor = $accent
    $button.FlatAppearance.MouseDownBackColor = $accent
    return $button
}

$form = New-Object System.Windows.Forms.Form
$form.Text = 'Select Python Environment'
$form.StartPosition = 'CenterScreen'
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$form.MinimizeBox = $false
$form.ClientSize = New-Object System.Drawing.Size(660, 235)
$form.BackColor = $background
$form.ForeColor = $foreground
$form.Font = New-Object System.Drawing.Font('Segoe UI', 10)

$form.Controls.Add((New-Label 'Select Python Environment' 20 15 620 26 $foreground $true))
$form.Controls.Add((New-Label 'Python executable (python.exe)' 20 52 620 22 $muted $false))
$box = New-Object System.Windows.Forms.TextBox
$box.Text = $initial
$box.Location = New-Object System.Drawing.Point(20, 76)
$box.Size = New-Object System.Drawing.Size(500, 26)
$box.BackColor = $fieldColor
$box.ForeColor = $foreground
$box.BorderStyle = 'FixedSingle'
$form.Controls.Add($box)
$browse = New-Button 'Browse...' 535 74 105
$form.Controls.Add($browse)
$form.Controls.Add((New-Label ('Python 3.10 or newer with Tkinter is required. The Python packages and ExifTool are checked ' +
    'when the window is opened (see the README to install them on a new computer).') 20 112 620 44 $muted $false))
$message = New-Label '' 20 160 620 22 $errorColor $false
$form.Controls.Add($message)
$confirm = New-Button 'Confirm' 430 190 100
$cancel = New-Button 'Cancel' 540 190 100
$form.Controls.Add($confirm)
$form.Controls.Add($cancel)
$form.AcceptButton = $confirm
$form.CancelButton = $cancel

$script:chosen = $null

$browse.Add_Click({
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = 'Select python.exe'
    $dialog.Filter = 'Python (python.exe)|python.exe|Programs (*.exe)|*.exe'
    $current = $box.Text.Trim().Trim('"')
    if ($current -and (Test-Path -LiteralPath (Split-Path -Parent $current) -ErrorAction SilentlyContinue)) {
        $dialog.InitialDirectory = Split-Path -Parent $current
    } elseif (Test-Path 'C:\path\to\your\envs') {
        $dialog.InitialDirectory = 'C:\path\to\your\envs'
    }
    if ($dialog.ShowDialog() -eq 'OK') { $box.Text = $dialog.FileName; $message.Text = '' }
})

$confirm.Add_Click({
    $path = $box.Text.Trim().Trim('"')
    $message.Text = 'Checking this Python...'
    $message.ForeColor = $muted
    $message.Refresh()
    $problem = Test-Python $path
    if ($problem) {
        $message.ForeColor = $errorColor
        $message.Text = $problem
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $file) | Out-Null
    (@{ python = $path } | ConvertTo-Json) | Set-Content -LiteralPath $file -Encoding UTF8
    $script:chosen = $path
    $form.DialogResult = 'OK'
    $form.Close()
})

[void]$form.ShowDialog()
if ($script:chosen) {
    Write-Output $script:chosen
    exit 0
}
exit 1
