# Increase Virtual Memory (Paging File) on Windows
# Run as Administrator

Write-Host "🔧 Setting up Virtual Memory (Paging File)..." -ForegroundColor Cyan

# Get system drive
$systemDrive = $env:SystemDrive

# Target size: 16GB in MB
$pageFileSizeMB = 16384

Write-Host "Configuring paging file on $systemDrive`:\"

# Get the current paging file settings
$pageFileSettings = Get-WmiObject Win32_PageFile

if ($null -eq $pageFileSettings) {
    Write-Host "No paging file found. Creating default..." -ForegroundColor Yellow
} else {
    Write-Host "Current paging file: $($pageFileSettings.Name)" -ForegroundColor Yellow
}

# Use registry to set paging file
$regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"

try {
    # Get current value
    $currentPageFile = Get-ItemProperty -Path $regPath -Name "PagingFiles" -ErrorAction SilentlyContinue
    
    if ($null -ne $currentPageFile) {
        Write-Host "Current setting: $($currentPageFile.PagingFiles)" -ForegroundColor Yellow
    }
    
    # Set new paging file: Drive:\pagefile.sys InitialSize MaxSize
    $pageFileValue = "$systemDrive\pagefile.sys $pageFileSizeMB $pageFileSizeMB"
    
    Set-ItemProperty -Path $regPath -Name "PagingFiles" -Value $pageFileValue
    
    Write-Host "✅ Paging file set to $pageFileSizeMB MB ($([math]::Round($pageFileSizeMB/1024, 2))GB)" -ForegroundColor Green
    Write-Host ""
    Write-Host "⚠️  PENTING: Silakan restart komputer untuk menerapkan perubahan!" -ForegroundColor Yellow
    Write-Host "Setelah restart, coba jalankan chatbot lagi." -ForegroundColor Yellow
    
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Alternatif manual:" -ForegroundColor Cyan
    Write-Host "1. Buka Settings > System > Advanced system settings" -ForegroundColor White
    Write-Host "2. Performance > Advanced tab" -ForegroundColor White
    Write-Host "3. Click 'Change' di Virtual Memory" -ForegroundColor White
    Write-Host "4. Uncheck 'Automatically manage paging file size'" -ForegroundColor White
    Write-Host "5. Set Initial size: 8192 MB (8GB)" -ForegroundColor White
    Write-Host "6. Set Max size: 16384 MB (16GB)" -ForegroundColor White
    Write-Host "7. Click OK dan restart" -ForegroundColor White
}
