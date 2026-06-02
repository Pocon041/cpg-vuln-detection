function Assert-EitEnvironment {
    if ($env:CONDA_DEFAULT_ENV -ne "EIT") {
        throw "Activate the expected environment first: conda activate EIT"
    }

    $actualPrefix = python -c "import sys; print(sys.prefix)"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect the active Python interpreter."
    }

    $actualEnvironment = Split-Path -Leaf $actualPrefix.Trim()
    if ($actualEnvironment -ne "EIT") {
        throw "Actual Python interpreter is not from EIT: $actualPrefix. Run: conda activate EIT"
    }
}

