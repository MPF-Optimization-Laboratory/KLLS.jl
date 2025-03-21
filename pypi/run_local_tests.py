#!/usr/bin/env python3
"""
Run tests against local Julia module

Options:
  --venv [DIR]   Create and use a virtual environment at optional DIR path
  --quiet        Suppress installation output messages
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run tests against local Julia module")
    parser.add_argument("--venv", nargs="?", const=True, default=False, 
                        help="Create and use a virtual environment at optional DIR path")
    parser.add_argument("--quiet", action="store_true", help="Suppress installation output messages")
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent.absolute()
    
    # Always use local Julia module
    os.environ["DUALPERSPECTIVE_USE_LOCAL"] = "true"
    
    # Change to the script directory
    os.chdir(script_dir)
    
    venv_dir = None
    try:
        if args.venv:
            # Create a virtual environment if needed
            if args.venv is True:  # No path specified, create a temp dir
                temp_dir = tempfile.mkdtemp()
                venv_dir = Path(temp_dir) / "venv"
            else:
                venv_dir = Path(args.venv)
            
            print(f"Creating virtual environment in {venv_dir}")
            venv.create(venv_dir, with_pip=True)
            
            # Get paths to binaries
            pip_path = venv_dir / "bin" / "pip"
            python_path = venv_dir / "bin" / "python"
            
            # Install dependencies
            stdout = subprocess.DEVNULL if args.quiet else None
            stderr = subprocess.DEVNULL if args.quiet else None
            
            subprocess.run(
                [str(pip_path), "install", "pytest"],
                check=True, stdout=stdout, stderr=stderr
            )
            subprocess.run(
                [str(pip_path), "install", "-e", "."],
                check=True, stdout=stdout, stderr=stderr
            )
            
            python_cmd = str(python_path)
        else:
            # Use system Python
            python_cmd = sys.executable
        
        # Configure pytest arguments
        pytest_args = ["tests/"]
        if args.quiet:
            pytest_args += ["-q", "--no-header"]
        else:
            pytest_args += ["-v"]
        
        # Run the tests
        result = subprocess.run(
            [python_cmd, "-m", "pytest"] + pytest_args,
            check=False
        )
        
        # Return the pytest exit code
        return result.returncode
        
    finally:
        # Clean up if we created a venv in a temp directory
        if venv_dir and args.venv is True:
            print(f"Cleaning up virtual environment")
            try:
                shutil.rmtree(venv_dir.parent, ignore_errors=True)
            except Exception as e:
                print(f"Failed to delete virtual environment at {venv_dir}: {e}")

if __name__ == "__main__":
    sys.exit(main()) 