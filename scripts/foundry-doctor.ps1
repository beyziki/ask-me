<#
.SYNOPSIS
    Foundry Local servisini kontrol eder, takılı kalmış (zombi) daemon'ı
    temizler ve servisi doğru yetki seviyesinde yeniden başlatır.

.DESCRIPTION
    Geliştirme sırasında tekrar tekrar karşılaşılan bir arıza var: Foundry
    Local daemon'ı (foundrylocald.exe) askıda kalıyor. `foundry server status`
    "Not running" diyor ama süreç hâlâ ayakta ve KULLANICIYA ÖZEL bir named
    pipe'ı tutmaya devam ediyor:

        foundry-cli-S-1-5-21-...-<kullanıcı>

    Bu yüzden her yeni daemon açılışta o pipe'ı alamayıp `AlreadyRunning (75)`
    koduyla anında çıkıyor; CLI bunu "Daemon did not start listening within
    15s" diye raporluyor. Gerçek sebep `foundry server logs` içinde görünür:

        DaemonBindContentionException: Another daemon already owns named pipe

    Bu betik o döngüyü kırar: takılı süreci bulur, önce CLI'ın kendi yoluyla
    (`foundry server stop`), olmazsa süreci sonlandırarak temizler, sonra
    servisi başlatıp gerçekten hazır olduğunu doğrular.

.NOTES
    YETKİ UYARISI — bu arızanın asıl kaynağı:
    Daemon YÜKSELTİLMİŞ (yönetici) bir terminalde başlatılırsa, normal
    yetkiyle çalışan backend onun named pipe'ına erişemez ve kendi daemon'ını
    başlatmaya çalışır — bu da yukarıdaki bind contention'a yol açar. Ayrıca
    yükseltilmiş süreci normal yetkiyle sonlandırmak da mümkün değildir
    ("Erişim engellendi").

    Bu yüzden: Foundry'i HER ZAMAN backend ile AYNI (normal) yetki seviyesinde
    çalıştır. Yönetici terminalini yalnızca takılı bir süreci öldürmek için
    kullan. Betik yönetici olarak çalıştırıldığında servisi başlatmayı
    reddeder ve yalnızca temizlik yapar.

.EXAMPLE
    .\scripts\foundry-doctor.ps1
    Durumu kontrol eder, gerekiyorsa temizler ve servisi başlatır.

.EXAMPLE
    .\scripts\foundry-doctor.ps1 -CleanOnly
    Yalnızca takılı daemon'ı temizler, servisi başlatmaz.
    (Yönetici terminalinde bunu kullan, sonra normal terminalde başlat.)
#>
[CmdletBinding()]
param(
    # Yalnızca temizlik yap, servisi başlatma.
    [switch]$CleanOnly,

    # Servisin hazır olması için beklenecek azami saniye. Soğuk başlangıçta
    # execution provider'ların (CUDA/WebGPU/OpenVINO/TensorRT) kaydı uzun
    # sürebildiği için cömert tutuldu.
    [int]$TimeoutSeconds = 90
)

$ErrorActionPreference = 'Stop'

function Test-IsElevated {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    return ([Security.Principal.WindowsPrincipal]$identity).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

function Get-DaemonProcesses {
    @(Get-Process foundrylocald -ErrorAction SilentlyContinue)
}

function Test-ServerReady {
    <# `foundry server status` çıktısında "Ready" arar. Metin tabanlı çünkü
       CLI'ın makine okunur bir çıktısı yok. #>
    $status = (& foundry server status 2>&1 | Out-String)
    return $status -match 'Ready'
}

$elevated = Test-IsElevated
Write-Host ''
Write-Host '=== Foundry Local doctor ===' -ForegroundColor Cyan
Write-Host ("Yetki seviyesi : {0}" -f $(if ($elevated) { 'YÖNETİCİ' } else { 'normal' }))

if ($elevated) {
    Write-Host ''
    Write-Host 'UYARI: Bu pencere yönetici yetkisiyle çalışıyor.' -ForegroundColor Yellow
    Write-Host 'Foundry burada BAŞLATILMAMALI — yükseltilmiş bir daemon, normal' -ForegroundColor Yellow
    Write-Host 'yetkili backend tarafından kullanılamaz. Temizlik yapılacak,' -ForegroundColor Yellow
    Write-Host 'servisi normal bir terminalde başlat.' -ForegroundColor Yellow
}

# --- 1) Mevcut durum ------------------------------------------------------
$processes = Get-DaemonProcesses
$ready = $false
try { $ready = Test-ServerReady } catch { $ready = $false }

Write-Host ("Daemon süreci  : {0}" -f $(if ($processes) { ($processes.Id -join ', ') } else { 'yok' }))
Write-Host ("Servis durumu  : {0}" -f $(if ($ready) { 'Ready' } else { 'hazır değil' }))

if ($ready -and -not $CleanOnly) {
    Write-Host ''
    Write-Host 'Servis zaten hazır, yapılacak bir şey yok.' -ForegroundColor Green
    if ($elevated) {
        Write-Host 'Ancak bu daemon yükseltilmiş bir oturumda başlatılmış olabilir;' -ForegroundColor Yellow
        Write-Host 'backend ona erişemezse -CleanOnly ile temizleyip normal' -ForegroundColor Yellow
        Write-Host 'terminalde yeniden başlat.' -ForegroundColor Yellow
    }
    exit 0
}

# --- 2) Takılı daemon'ı temizle ------------------------------------------
# "Süreç var ama servis hazır değil" tam olarak zombi durumudur: named pipe
# tutuluyor, yeni daemon bind edemiyor.
if ($processes) {
    Write-Host ''
    Write-Host 'Takılı daemon tespit edildi, temizleniyor...' -ForegroundColor Yellow

    # Önce CLI'ın kendi yolu — en temizi.
    try { & foundry server stop 2>&1 | Out-Null } catch { }
    Start-Sleep -Seconds 2

    foreach ($proc in Get-DaemonProcesses) {
        try {
            Stop-Process -Id $proc.Id -Force -ErrorAction Stop
            Write-Host ("  PID {0} sonlandırıldı." -f $proc.Id) -ForegroundColor Green
        } catch {
            Write-Host ("  PID {0} sonlandırılamadı: {1}" -f $proc.Id, $_.Exception.Message) -ForegroundColor Red
            Write-Host ''
            Write-Host 'Bu süreç büyük olasılıkla yönetici yetkisiyle başlatılmış.' -ForegroundColor Red
            Write-Host 'Yönetici bir PowerShell açıp şunu çalıştır:' -ForegroundColor Red
            Write-Host ("  .\scripts\foundry-doctor.ps1 -CleanOnly") -ForegroundColor Red
            Write-Host 'Sonra bu betiği normal terminalde tekrar çalıştır.' -ForegroundColor Red
            exit 1
        }
    }

    # Süreç modeli bellekte tutuyorsa tamamen kapanması birkaç saniye sürebilir;
    # named pipe serbest kalmadan yeni daemon yine bind edemez.
    for ($i = 0; $i -lt 15; $i++) {
        if (-not (Get-DaemonProcesses)) { break }
        Start-Sleep -Seconds 1
    }

    if (Get-DaemonProcesses) {
        Write-Host 'Süreç hâlâ kapanmadı. Birkaç saniye bekleyip tekrar dene.' -ForegroundColor Red
        exit 1
    }
    Write-Host '  Temiz.' -ForegroundColor Green
}
else {
    Write-Host ''
    Write-Host 'Takılı süreç yok.' -ForegroundColor Green
}

if ($CleanOnly) {
    Write-Host ''
    Write-Host 'Temizlik tamam. Servisi NORMAL bir terminalde başlat:' -ForegroundColor Cyan
    Write-Host '  .\scripts\foundry-doctor.ps1'
    exit 0
}

if ($elevated) {
    Write-Host ''
    Write-Host 'Yönetici penceresinde olduğumuz için servis BAŞLATILMIYOR.' -ForegroundColor Yellow
    Write-Host 'Bu pencereyi kapat, normal bir terminalde bu betiği tekrar çalıştır.' -ForegroundColor Yellow
    exit 0
}

# --- 3) Servisi başlat ----------------------------------------------------
Write-Host ''
Write-Host 'Servis başlatılıyor...' -ForegroundColor Cyan
& foundry server start

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    if (Test-ServerReady) {
        Write-Host ''
        Write-Host 'Foundry Local hazır.' -ForegroundColor Green
        & foundry server status
        Write-Host ''
        Write-Host 'Sıradaki adımlar:' -ForegroundColor Cyan
        Write-Host '  foundry model list       # modelin Cached (●) olduğunu doğrula'
        Write-Host '  uvicorn backend.app.main:app --reload --reload-dir backend'
        exit 0
    }
    Start-Sleep -Seconds 2
}

Write-Host ''
Write-Host ("Servis {0} saniyede hazır olmadı." -f $TimeoutSeconds) -ForegroundColor Red
Write-Host 'Ayrıntı için: foundry server logs' -ForegroundColor Red
Write-Host '"DaemonBindContentionException" görüyorsan hâlâ takılı bir süreç var.' -ForegroundColor Red
exit 1
