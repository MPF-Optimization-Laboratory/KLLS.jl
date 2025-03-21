# Optional Python bindings test
using Test
using Random

# Check for necessary Python tools
python_cmd = Sys.which("python3") !== nothing ? Sys.which("python3") : Sys.which("python")
if python_cmd === nothing
    @warn "Python not found, skipping Python binding tests"
    return
end

# Path to the script and pypi directory
pypi_dir = joinpath(dirname(@__DIR__), "pypi")
run_tests_script = joinpath(pypi_dir, "run_local_tests.py")

# Make sure the script is executable
chmod(run_tests_script, 0o755)

# Run the test script with parameters that match our original behavior:
# - Use a temporary virtual environment
# - Run in quiet mode to minimize output
# - Script will exit non-zero on failure
temp_venv_dir = joinpath(tempdir(), "dp_test_venv_$(randstring(8))")

# Run the tests via the Python script directly
test_cmd = `$(python_cmd) $(run_tests_script) --venv $(temp_venv_dir) --quiet`

# Capture output and error
output = IOBuffer()
error_output = IOBuffer()

# Run tests
@testset "Python Bindings" begin
    try
        process = run(pipeline(test_cmd, stdout=output, stderr=error_output))
        @test process.exitcode == 0
    catch e
        # If process failed to start or was interrupted
        if isa(e, ProcessFailedException)
            # Show captured outputs to help with debugging
            seekstart(output)
            seekstart(error_output)
            println("Standard output:")
            println(String(take!(output)))
            println("Standard error:")
            println(String(take!(error_output)))
            @test false
        else
            rethrow(e)
        end
    end
end