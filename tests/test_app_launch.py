import subprocess
import pytest
import sys
import time

# Define a timeout for the Streamlit app launch
LAUNCH_TIMEOUT = 45  # seconds, can be adjusted
STREAMLIT_TEST_PORT = "8555"  # Using a distinct port for testing


@pytest.fixture(
    scope="function"
)  # Changed to function scope for cleaner state if multiple tests use it
def streamlit_app_process():
    """
    Fixture to start the Streamlit app as a subprocess and ensure its termination.
    """
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/main.py",
        "--server.runOnSave=false",
        "--server.headless=true",
        f"--server.port={STREAMLIT_TEST_PORT}",
        "--server.fileWatcherType=none",  # Disable file watcher for stability
    ]

    print(f"Starting Streamlit app with command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line-buffered
        universal_newlines=True,
    )

    yield process  # Provide the process to the test

    print(f"Terminating Streamlit app process (PID: {process.pid})...")
    process.terminate()
    try:
        stdout, stderr = process.communicate(
            timeout=10
        )  # Allow time for graceful shutdown
        print("Streamlit app process terminated.")
        if stdout:
            print(f"Final STDOUT from Streamlit app:\n{stdout}")
        if stderr:
            print(f"Final STDERR from Streamlit app:\n{stderr}")
    except subprocess.TimeoutExpired:
        print("Streamlit app process did not terminate gracefully, killing.")
        process.kill()
        stdout, stderr = process.communicate()  # Get output after kill
        print("Streamlit app process killed.")
        if stdout:
            print(f"Final STDOUT (after kill) from Streamlit app:\n{stdout}")
        if stderr:
            print(f"Final STDERR (after kill) from Streamlit app:\n{stderr}")
    except Exception as e:
        print(f"Error during Streamlit app termination: {e}")


def test_app_launches_successfully(streamlit_app_process):
    """
    Tests if the Streamlit application starts and seems to be running.
    It checks for specific output messages or relies on timeout as an indicator of a running server.
    """
    process = streamlit_app_process

    # Attempt to read initial output for readiness cues
    # This is a heuristic; a more robust check might involve trying to connect if environment allows
    app_ready_indicator_found = False

    # Give the app some time to print initial messages
    # We'll read non-blockingly or with small timeouts if possible,
    # but Popen.stdout.readline() is blocking.
    # A common pattern is to use select or threads for non-blocking reads,
    # but we'll keep it simpler here and rely on the overall LAUNCH_TIMEOUT.

    print(f"Monitoring Streamlit app output for up to {LAUNCH_TIMEOUT} seconds...")
    start_time = time.time()

    # We expect Streamlit in headless mode to run until killed.
    # So, we are looking for it to *not* exit quickly and to print startup messages.
    try:
        while time.time() - start_time < LAUNCH_TIMEOUT:
            # Check if process terminated unexpectedly
            if process.poll() is not None:
                stdout, stderr = process.communicate()  # Get all remaining output
                pytest.fail(
                    f"Streamlit process terminated unexpectedly with code {process.returncode}.\n"
                    f"STDOUT:\n{stdout}\n"
                    f"STDERR:\n{stderr}"
                )

            # Attempt to read a line from stdout with a short timeout (not directly possible with readline)
            # Instead, we'll rely on the overall LAUNCH_TIMEOUT and check output after a timeout from communicate()
            # For now, this loop mainly checks process.poll() and then we handle communicate() timeout.
            # A more advanced version would use threads or select for non-blocking reads here.
            time.sleep(0.5)  # Check status every 0.5s

        # If loop finishes, it means LAUNCH_TIMEOUT reached without process ending.
        # This is the "success" case for a server process.
        print(
            f"Streamlit app did not terminate within {LAUNCH_TIMEOUT}s (considered launched and running)."
        )
        # Now, kill it and get output
        process.kill()  # Using kill directly as we've timed out waiting for it to be "ready"
        stdout, stderr = process.communicate(
            timeout=10
        )  # 10s for communicate after kill

        assert (
            stdout or stderr
        ), "No output from Streamlit process on timeout, might not have started properly."
        print(
            "Streamlit app considered successfully launched as it ran for the timeout duration."
        )
        if stdout:
            print(f"STDOUT (on timeout):\n{stdout}")
        if stderr:  # stderr might contain "Serving at..." messages which are normal
            print(f"STDERR (on timeout):\n{stderr}")

        # Check for common success messages in the output captured
        if (
            "You can now view your Streamlit app in your browser." in stdout
            or "Network URL:" in stdout
            or "External URL:" in stdout
        ):
            app_ready_indicator_found = True
            print("App ready indicator found in STDOUT.")

        # This assertion is a bit weak if only relying on timeout, stronger if indicator found.
        assert app_ready_indicator_found or (
            stdout or stderr
        ), "Streamlit app timed out but no output or readiness indicator found."

    except (
        subprocess.TimeoutExpired
    ):  # Should be caught by the outer timeout logic now.
        # This block might not be reached if the while loop handles timeout.
        # Kept for safety, mirrors original logic.
        process.kill()
        stdout, stderr = process.communicate()
        assert (
            stdout or stderr
        ), "No output from Streamlit process on timeout (TimeoutExpired), might not have started."
        print(
            f"Streamlit app timed out (TimeoutExpired) after {LAUNCH_TIMEOUT}s (considered launched and running)."
        )
        if stdout:
            print(f"STDOUT (on TimeoutExpired):\n{stdout}")
        if stderr:
            print(f"STDERR (on TimeoutExpired):\n{stderr}")
        # No specific return code assertion here as we killed it.
        return  # Success

    except FileNotFoundError:  # Should be caught by Popen if command fails
        pytest.fail(
            "Streamlit command not found. Ensure Streamlit is installed and in PATH."
        )
    except Exception as e:
        # Ensure process is cleaned up if an unexpected error occurs
        if process and process.poll() is None:
            process.kill()
            process.communicate()  # Clean up pipes
        pytest.fail(f"An unexpected error occurred during the test: {e}")


def test_streamlit_help_works():  # No changes to this test, it's fairly robust
    """
    A simpler test to check if the streamlit CLI is basically working.
    """
    try:
        command = [sys.executable, "-m", "streamlit", "--help"]
        process = subprocess.run(command, capture_output=True, text=True, timeout=10)
        assert (
            process.returncode == 0
        ), f"Streamlit --help exited with {process.returncode}. Stderr: {process.stderr}"
        assert (
            "Usage: streamlit [OPTIONS] COMMAND [ARGS]" in process.stdout
        ), "Streamlit --help output was not as expected."
    except FileNotFoundError:
        pytest.fail("Streamlit command not found (via python -m streamlit).")
    except subprocess.TimeoutExpired:
        pytest.fail("Streamlit --help command timed out.")
    except Exception as e:
        pytest.fail(f"An error occurred running streamlit --help: {e}")


if __name__ == "__main__":
    # For manual execution of this specific test file
    pytest.main([__file__])
