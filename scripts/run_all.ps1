param(
    [string]$Config = "configs/default.yaml",
    [switch]$SkipCodeBert,
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\assert_eit.ps1"
Assert-EitEnvironment

python -m cpg_vuln --config $Config audit
python -m cpg_vuln --config $Config build-topologies
python -m cpg_vuln --config $Config build-word2vec

if (-not $SkipCodeBert) {
    python -m cpg_vuln --config $Config build-codebert-cache
}

if (-not $SkipTraining) {
    if ($SkipCodeBert) {
        python -m cpg_vuln --config $Config train-baselines --embeddings word2vec
    } else {
        python -m cpg_vuln --config $Config train-baselines
    }
    if (-not $SkipCodeBert) {
        python -m cpg_vuln --config $Config train-enhanced
    }
    python -m cpg_vuln --config $Config summarize
    if (-not $SkipCodeBert) {
        python -m cpg_vuln --config $Config explain
    }
}
