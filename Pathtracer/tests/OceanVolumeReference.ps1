# Scalar transport checks only. Does not compile shaders, build, or launch the renderer.
$ErrorActionPreference = 'Stop'
$script:checks = 0
function Near([double]$actual, [double]$expected, [double]$tolerance, [string]$label) {
    if ([double]::IsNaN($actual) -or [double]::IsInfinity($actual) -or
        [Math]::Abs($actual-$expected) -gt $tolerance) { throw "$label : $actual != $expected" }
    $script:checks++
}
function Mass([double]$rate, [double]$distance) {
    $x = $rate*$distance
    if ($x -lt 0.001) { return $x*(1-$x/2+$x*$x/6) }
    return 1-[Math]::Exp(-$x)
}
function SampleDistance([double]$rate, [double]$distance, [double]$u) {
    if ($rate*$distance -lt 0.001) { return $u*$distance }
    return -[Math]::Log([Math]::Max(1-$u*(Mass $rate $distance),1e-7))/$rate
}
function Pdf([double]$rate, [double]$distance, [double]$t) {
    if ($rate*$distance -lt 0.001) { return 1/$distance }
    return $rate*[Math]::Exp(-$rate*$t)/(Mass $rate $distance)
}
foreach ($f in @(0.0, 0.0203731878419714, 0.15, 0.7, 0.95, 1.0)) {
    $q = if ($f -eq 0) { 0.0 } elseif ($f -eq 1) { 1.0 } else { [Math]::Max(0.7,$f) }
    $reflection = if ($q -gt 0) { $q*($f/$q) } else { 0.0 }
    $transmission = if ($q -lt 1) { (1-$q)*((1-$f)/(1-$q)) } else { 0.0 }
    Near $reflection $f 1e-12 'Proposal-independent reflected energy'
    Near $transmission (1-$f) 1e-12 'Proposal-independent transmitted energy'
}
foreach ($rate in @(0.0, 0.00001, 0.01, 0.1, 1.0, 10.0)) {
    foreach ($distance in @(0.0, 0.001, 0.5, 10.0, 100.0)) {
        $t = [Math]::Exp(-$rate*$distance)
        Near ([Math]::Exp(-$rate*$distance*0.3)*[Math]::Exp(-$rate*$distance*0.7)) $t 1e-12 'Segment composition'
        if ($distance -eq 0) { continue }
        foreach ($u in @(0.0,0.1,0.5,0.9,0.999)) {
            $sample = SampleDistance $rate $distance $u
            if ($sample -lt 0 -or $sample -gt $distance+1e-9 -or (Pdf $rate $distance $sample) -le 0) { throw 'Invalid distance sample' }
            $script:checks++
        }
    }
}
# Deterministic quadrature over the RGB mixture proposals. Constant incident source
# has closed-form integral (1-exp(-sigma_t*d))/sigma_t in each channel.
$rates = @(0.02,0.1,0.8)
$steps = 4096
$distance = 30.0
$integrals = @(0.0,0.0,0.0)
foreach ($proposal in $rates) {
    for ($i=0; $i -lt $steps; $i++) {
        $t = SampleDistance $proposal $distance (($i+0.5)/$steps)
        $pdf = ((Pdf $rates[0] $distance $t)+(Pdf $rates[1] $distance $t)+(Pdf $rates[2] $distance $t))/3
        for ($c=0; $c -lt 3; $c++) { $integrals[$c] += [Math]::Exp(-$rates[$c]*$t)/($pdf*3*$steps) }
    }
}
for ($c=0; $c -lt 3; $c++) {
    Near $integrals[$c] ((1-[Math]::Exp(-$rates[$c]*$distance))/$rates[$c]) 0.002 'RGB distance estimator'
}
foreach ($g in @(-0.95,-0.5,0.0,0.5,0.95)) {
    $sum = 0.0
    for ($i=0; $i -lt 100000; $i++) {
        $mu = -1+2*($i+0.5)/100000
        $d = 1+$g*$g-2*$g*$mu
        $sum += (1-$g*$g)/(2*$d*[Math]::Sqrt($d))*2/100000
    }
    Near $sum 1.0 0.00005 'HG normalization'
}
Write-Output "$script:checks scalar checks passed. No build or visual test performed."
