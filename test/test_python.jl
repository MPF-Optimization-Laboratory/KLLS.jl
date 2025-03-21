# Optional Python bindings test
using Test
using Random

# Check for necessary Python tools
python_cmd = Sys.which("python3") !== nothing ? Sys.which("python3") : Sys.which("python")
if python_cmd === nothing
    @warn "Python not found, skipping Python binding tests"
    return
end

pypi_dir = joinpath(dirname(@__DIR__), "pypi")

# Create a temporary virtual environment
temp_dir = joinpath(tempdir(), "dp_test_venv_$(randstring(8))")
venv_dir = joinpath(temp_dir, "venv")
venv_bin_dir = joinpath(venv_dir, "bin")
venv_python = joinpath(venv_bin_dir, "python")
venv_pip = joinpath(venv_bin_dir, "pip")
venv_pytest = joinpath(venv_bin_dir, "pytest")

try
    # Create the virtual environment
    run(pipeline(`$(python_cmd) -m venv $(venv_dir)`, stdout=devnull, stderr=devnull))
    
    # Install the package in development mode with local flag
    withenv("DUALPERSPECTIVE_USE_LOCAL" => "true") do
        # Install pytest and current package
        install_pip_cmd = pipeline(`$(venv_pip) install -q pytest`, stdout=devnull, stderr=devnull)
        run(install_pip_cmd)
        
        install_pkg_cmd = pipeline(`$(venv_pip) install -q -e $(pypi_dir)`, stdout=devnull, stderr=devnull)
        run(install_pkg_cmd)
        
        # Run the tests using the virtualenv pytest - only show output on failure
        test_cmd = pipeline(`$(venv_pytest) $(pypi_dir)/tests -q --no-header`, stdout=devnull, stderr=devnull)
        @test success(run(test_cmd))
    end
finally
    # Clean up temporary directory
    try
        rm(temp_dir, recursive=true, force=true)
    catch
        @warn "Failed to clean up temporary directory: $temp_dir"
    end
end