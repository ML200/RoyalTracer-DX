# Source/scalar verification only: no compiler, renderer or visual test.
$ErrorActionPreference='Stop'
$script:checks=0
function Near([double]$actual,[double]$expected,[double]$tolerance,[string]$label) {
    if ([double]::IsNaN($actual) -or [double]::IsInfinity($actual) -or [Math]::Abs($actual-$expected) -gt $tolerance) {
        throw "$label : $actual != $expected"
    }
    $script:checks++
}
function Flight($a,$s,[double]$limit,[bool]$used,[int]$channel,[double]$u) {
    $rates=@(($a[0]+$s[0]),($a[1]+$s[1]),($a[2]+$s[2]))
    $event=$false
    $t=$limit
    $randomFlight=(!$used -and ($s[0]+$s[1]+$s[2]) -gt 0 -and $limit -gt 0)
    if ($randomFlight) {
        $sample=if($rates[$channel] -gt 0){-[Math]::Log([Math]::Max(1-$u,1e-7))/$rates[$channel]}else{$limit}
        $event=$sample -lt $limit
        $t=[Math]::Min($sample,$limit)
    }
    $tr=@([Math]::Exp(-$rates[0]*$t),[Math]::Exp(-$rates[1]*$t),[Math]::Exp(-$rates[2]*$t))
    $weight=@($tr[0],$tr[1],$tr[2])
    if ($randomFlight) {
        $pdf=if($event){($rates[0]*$tr[0]+$rates[1]*$tr[1]+$rates[2]*$tr[2])/3}else{($tr[0]+$tr[1]+$tr[2])/3}
        for($c=0;$c -lt 3;$c++) {$weight[$c]=if($pdf -gt 0){$tr[$c]*$(if($event){$s[$c]}else{1})/$pdf}else{0}}
    }
    return @{distance=$t; weight=$weight; scattered=$event}
}
foreach($f in @(0.0,0.0203731878419714,0.15,0.7,0.95,1.0)) {
    $q=if($f -eq 0){0.0}elseif($f -eq 1){1.0}else{[Math]::Max(0.7,$f)}
    Near $(if($q -gt 0){$q*($f/$q)}else{0}) $f 1e-12 'Fresnel reflected energy'
    Near $(if($q -lt 1){(1-$q)*((1-$f)/(1-$q))}else{0}) (1-$f) 1e-12 'Fresnel transmitted energy'
}
# Both estimator branches vs analytic homogeneous transport.
$cases=@(@{a=@(0.1,0.01,0.4);s=@(0.05,0.2,0.03)},@{a=@(0.0,0.05,0.3);s=@(0.0,0.1,0.2)})
$steps=4096
foreach($case in $cases) {
    foreach($distance in @(0.01,2.0,30.0)) {
        $endpoint=@(0.0,0.0,0.0); $scatter=@(0.0,0.0,0.0)
        for($channel=0;$channel -lt 3;$channel++) {
            for($i=0;$i -lt $steps;$i++) {
                $f=Flight $case.a $case.s $distance $false $channel (($i+0.5)/$steps)
                for($c=0;$c -lt 3;$c++) {
                    if($f.scattered){$scatter[$c]+=$f.weight[$c]/(3*$steps)}else{$endpoint[$c]+=$f.weight[$c]/(3*$steps)}
                }
            }
        }
        for($c=0;$c -lt 3;$c++) {
            $rate=$case.a[$c]+$case.s[$c]; $tr=[Math]::Exp(-$rate*$distance)
            Near $endpoint[$c] $tr 0.0008 'Surface branch including no-event PDF'
            $expected=if($rate -gt 0){$case.s[$c]/$rate*(1-$tr)}else{0}
            Near $scatter[$c] $expected 0.0008 'Scattering branch including event PDF'
        }
    }
}
foreach($used in @($false,$true)) {
    foreach($distance in @(0.0,0.1,10.0,1000.0)) {
        $f=Flight @(0.1,0.2,0.3) @(0.0,0.0,0.0) $distance $used 1 0.1
        if($f.scattered){throw 'Absorption-only medium scattered'}
        for($c=0;$c -lt 3;$c++){Near $f.weight[$c] ([Math]::Exp(-0.1*($c+1)*$distance)) 1e-12 'Absorption only'}
    }
}
$f=Flight @(0.1,0.1,0.1) @(0.2,0.2,0.2) 2 $true 0 0.01
if($f.scattered){throw 'Second event allowed'}
Near $f.weight[0] ([Math]::Exp(-0.6)) 1e-12 'Extinction after the one scatter'
foreach($g in @(-0.95,-0.5,0.0,0.5,0.95)) {
    $mean=0.0; $square=0.0
    for($i=0;$i -lt 16384;$i++) {
        $u=($i+0.5)/16384
        $mu=if([Math]::Abs($g) -lt 0.001){1-2*$u}else{(1+$g*$g-[Math]::Pow((1-$g*$g)/(1-$g+2*$g*$u),2))/(2*$g)}
        $mean+=$mu/16384; $square+=$mu*$mu/16384
    }
    Near $mean $g 0.00001 'HG sampled mean'
    Near $square ((1+2*$g*$g)/3) 0.00001 'HG sampled second moment'
}
# Scatter budget: kept through internal reflection and thin panes, reset by bounces and entries.
$stateCases=@(
    @($true,$true,$true,$true,$false,$true),
    @($true,$true,$true,$false,$false,$false),
    @($true,$true,$true,$false,$true,$true),
    @($true,$false,$true,$true,$false,$false),
    @($true,$true,$false,$true,$false,$false),
    @($false,$true,$true,$true,$false,$false))
foreach($c in $stateCases){Near ([int]($c[0] -and $c[1] -and $c[2] -and ($c[3] -or $c[4]))) ([int]$c[5]) 0 'Event reset state'}
$root=Split-Path $PSScriptRoot -Parent
$volume=Get-Content (Join-Path $root 'shaders/OceanVolume.hlsli') -Raw
if($volume -match 'OceanHeight\(|OceanIntegrateVolume'){throw 'Expensive volume integration remains'}
$camera=Get-Content (Join-Path $root 'shaders/Pass_camera_v8.hlsl') -Raw
$trace=Get-Content (Join-Path $root 'shaders/Pass_pt_trace_v8.hlsl') -Raw
$training=Get-Content (Join-Path $root 'shaders/Pass_sharc_update_v8.hlsl') -Raw
$resolve=Get-Content (Join-Path $root 'shaders/Pass_shading_v8.hlsl') -Raw
if(([regex]::Matches($camera,'OceanPointInside\(')).Count -ne 1 -or ($trace+$training+$resolve) -match 'OceanPointInside\('){throw 'Repeated camera classification'}
if($trace -notmatch 'SD_FLAG_NOBOUNCE\) != 0u && !cameraWater' -or $trace -notmatch 'PV_IN_WATER_SCATTERED' -or $training -notmatch 'waterScatterUsed = true'){throw 'Path state wiring missing'}
if($resolve -match 'OceanIntegrateVolume|OceanSampleWaterSegment'){throw 'Duplicate camera volume'}
# Driver workaround: no HitObject carried through the scattering loops.
$callerCode=[regex]::Replace(($trace+$training),'(?s)/\*.*?\*/|//[^\r\n]*','')
if($callerCode -match 'dx::HitObject|hitObj\.' -or $volume -notmatch 'OceanPathHit OceanTracePathHit\(RayDesc ray\)') {
    throw 'Opaque hit-object lifetime regression'
}
foreach($getter in @('GetRayTCurrent','GetInstanceID','GetGeometryIndex','GetPrimitiveIndex','GetAttributes')) {
    if($volume -notmatch $getter){throw "Trace snapshot lost $getter"}
}
Write-Output "$script:checks scalar checks and 5 source contracts passed. No build or visual test performed."
