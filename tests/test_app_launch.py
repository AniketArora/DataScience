import subprocess
import pytest
import sys # To get python interpreter path

# Define a timeout for the Streamlit app launch
LAUNCH_TIMEOUT = 30  # seconds

def test_app_launches_successfully():
    """
    Tests if the Streamlit application launches successfully within a timeout.
    It checks for a return code of 0 (success) or 1 (expected if already running or port issue).
    It also verifies that there is some output to stdout or stderr.
    """
    try:
        # Command to run Streamlit in headless mode for testing
        # --server.runOnSave=false is important to prevent continuous reruns in some environments
        # --server.headless=true is crucial for CI environments
        # --server.port can be set to a specific free port if default 8501 causes issues
        command = [
            sys.executable, "-m", "streamlit", "run", "src/main.py",
            "--server.runOnSave=false",
            "--server.headless=true"
        ]

        # Start the Streamlit app as a subprocess
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Decode stdout/stderr as text
        )

        # Wait for the process to complete or timeout
        # Streamlit in headless mode might run until explicitly killed.
        # We are looking for it to start up and print initial messages.
        # A short sleep might be needed for Streamlit to initialize before we terminate.
        try:
            stdout, stderr = process.communicate(timeout=LAUNCH_TIMEOUT)
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()  # Ensure the process is killed if it times out
            stdout, stderr = process.communicate()
            # If it times out, it means Streamlit started and was running.
            # This is a successful launch for a server application.
            # We don't expect Streamlit to exit on its own in headless mode quickly.
            # Consider timeout as a sign of successful persistent server start.
            # However, for a quick "does it start" check, we might expect it to error out if misconfigured.
            # For now, let's assume timeout means it's running.
            # A more robust check would be to query the Streamlit health endpoint if available.
            # For this test, we'll consider a timeout as "launched and running".
            # If Streamlit exits very quickly with 0, it might not have fully started.
            # If it errors out quickly (non-0), that's a failure.

            # Check if there's any output, which indicates it tried to start
            assert stdout or stderr, "No output from Streamlit process on timeout, might not have started."
            # If it timed out, it means it was running. We can consider this a pass for "launches".
            # No specific return code assertion here as we killed it.
            print(f"Streamlit app timed out after {LAUNCH_TIMEOUT}s (considered launched and running).")
            print(f"STDOUT:\n{stdout}")
            print(f"STDERR:\n{stderr}")
            return # Success case for timeout

        # If communicate() completed without timeout (Streamlit exited on its own)
        # This might happen if there's a critical startup error.
        # A normal Streamlit server launch will not exit quickly unless --server.headless=true
        # has a different behavior or an immediate error occurs.

        # Check the return code
        # Streamlit might return 0 if it starts and then something immediately stops it (e.g. if it's not a "serve" command)
        # or if --server.headless=true has a mode where it exits after confirming it *can* start.
        # It might return 1 for common issues like port already in use.
        # For a simple "can it start?" test, any output is good.
        # A more robust test would involve checking a health endpoint.
        assert return_code == 0, f"Streamlit process exited with code {return_code}.\nStdout: {stdout}\nStderr: {stderr}"

        # Check that there was some output
        assert stdout or stderr, "No output from Streamlit process, it might not have started correctly."

        print(f"Streamlit app launch test completed with return code {return_code}.")
        if stdout:
            print(f"STDOUT:\n{stdout}")
        if stderr:
            print(f"STDERR:\n{stderr}")

    except FileNotFoundError:
        pytest.fail("Streamlit command not found. Ensure Streamlit is installed and in PATH.")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during the test: {e}")

def test_streamlit_help_works():
    """
    A simpler test to check if the streamlit CLI is basically working.
    """
    try:
        command = [sys.executable, "-m", "streamlit", "--help"]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10
        )
        assert process.returncode == 0, f"Streamlit --help exited with {process.returncode}. Stderr: {process.stderr}"
        assert "Usage: streamlit [OPTIONS] COMMAND [ARGS]" in process.stdout, "Streamlit --help output was not as expected."
    except FileNotFoundError:
        pytest.fail("Streamlit command not found (via python -m streamlit).")
    except subprocess.TimeoutExpired:
        pytest.fail("Streamlit --help command timed out.")
    except Exception as e:
        pytest.fail(f"An error occurred running streamlit --help: {e}")

if __name__ == "__main__":
    # For manual execution of this specific test file
    pytest.main([__file__])
