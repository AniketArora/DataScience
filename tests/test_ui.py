import pytest
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

# Define a timeout for the Streamlit app launch and page load
APP_LAUNCH_TIMEOUT = (
    90  # Increased seconds for Streamlit to start (Not used in this simplified version)
)
PAGE_LOAD_TIMEOUT = 30  # Increased seconds for page elements to appear (Not used in this simplified version)
STREAMLIT_APP_URL = "http://localhost:8501"  # Default Streamlit port (Not used in this simplified version)

# @pytest.fixture(scope="module")
# def streamlit_server():
#     """
#     Fixture to start and stop the Streamlit server for the test module.
#     Temporarily disabled to isolate WebDriver issues.
#     """
#    command = [
#        sys.executable, "-m", "streamlit", "run", "src/main.py",
#        "--server.runOnSave=false",
#        "--server.headless=true",
#        "--server.port=8501", # Ensure a consistent port
#        "--server.fileWatcherType=none" # Disable file watcher for stability in tests
#    ]
#
#    print(f"Starting Streamlit server with command: {' '.join(command)}")
#    server_process = subprocess.Popen(
#        command,
#        stdout=subprocess.PIPE,
#        stderr=subprocess.PIPE,
#        text=True,
#        bufsize=1,  # Line buffered
#        universal_newlines=True # Ensure text mode works correctly
#    )
#
#    # Wait for Streamlit to start by checking for a common startup message
#    start_time = time.time()
#    app_ready = False
#
#    print("Waiting for Streamlit server to become ready...")
#    log_buffer = ""
#
#    # Simplified readiness check: Wait a fixed time and then try to connect.
#    # This avoids complex log parsing which might be problematic.
#    print(f"Giving Streamlit server {APP_LAUNCH_TIMEOUT // 3} seconds to initialize before attempting connection checks...")
#    time.sleep(APP_LAUNCH_TIMEOUT // 3) # Wait for a portion of the timeout initially.
#
#    # Try to connect to the server to confirm it's up
#    # This is a more direct way to check than just log parsing
#    import socket
#    app_ready = False
#    for i in range(3): # Try a few times
#        try:
#            with socket.create_connection((STREAMLIT_APP_URL.split(":")[1].replace("//",""), int(STREAMLIT_APP_URL.split(":")[2])), timeout=10):
#                print(f"Successfully connected to Streamlit server at {STREAMLIT_APP_URL}.")
#                app_ready = True
#                break
#        except (socket.error, ConnectionRefusedError) as e:
#            print(f"Connection attempt {i+1}/3 to Streamlit server failed: {e}. Waiting and retrying...")
#            time.sleep(5) # Wait 5 seconds before retrying
#
#    if not app_ready:
#        print(f"Failed to connect to Streamlit server at {STREAMLIT_APP_URL} after multiple attempts.")
#        # Try to dump any output from the server process if it terminated or has output
#        stdout_output = ""
#        stderr_output = ""
#        try:
#            # Non-blocking read of what's available
#            # This is tricky with subprocess.PIPE without select or threads.
#            # communicate() will wait for process to end, which we don't want yet if it's just slow.
#            # For simplicity in this step, we'll rely on the terminate/kill below and its communicate() call.
#            if server_process.poll() is not None: # If process already terminated
#                 stdout_output, stderr_output = server_process.communicate(timeout=5)
#            else: # Process still running, but we deemed it not ready
#                 pass # We'll get output during teardown
#        except Exception as e:
#            print(f"Error trying to get initial output from non-ready server: {e}")
#
#        server_process.terminate()
#        try:
#            stdout_after_terminate, stderr_after_terminate = server_process.communicate(timeout=10)
#            stdout_output += stdout_after_terminate
#            stderr_output += stderr_after_terminate
#        except subprocess.TimeoutExpired:
#            server_process.kill()
#            stdout_after_kill, stderr_after_kill = server_process.communicate(timeout=10)
#            stdout_output += stdout_after_kill
#            stderr_output += stderr_after_kill
#        except Exception as e:
#             print(f"Error during server termination communication: {e}")
#
#
#        print(f"Streamlit server STDOUT:\n{stdout_output}")
#        print(f"Streamlit server STDERR:\n{stderr_output}")
#        pytest.fail(f"Streamlit server failed to start and become connectable at {STREAMLIT_APP_URL} within the allotted time.")
#
#    print("Streamlit server process appears to be running and connectable.")
#    yield server_process
#
#    print("Tearing down Streamlit server...")
#    server_process.terminate()
#    try:
#        server_process.wait(timeout=10)
#    except subprocess.TimeoutExpired:
#        print("Streamlit server did not terminate gracefully, killing.")
#        server_process.kill()
#
#    stdout, stderr = server_process.communicate()
#    print(f"Streamlit server STDOUT after termination:\n{stdout}")
#    print(f"Streamlit server STDERR after termination:\n{stderr}")
# #     print("Streamlit server torn down.")


@pytest.fixture(scope="function")
# def browser(streamlit_server): # Temporarily remove streamlit_server dependency
def browser():  # No streamlit_server dependency
    """
    Fixture to initialize and quit the Selenium WebDriver for each test function.
    """
    print("Setting up Selenium WebDriver...")
    # Setup Chrome options for headless execution if needed (e.g., in CI)
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")  # Run headless
    chrome_options.add_argument(
        "--no-sandbox"
    )  # Bypass OS security model, REQUIRED for Docker/CI
    chrome_options.add_argument(
        "--disable-dev-shm-usage"
    )  # Overcome limited resource problems
    chrome_options.add_argument("--disable-gpu")  # Applicable to windows os only
    chrome_options.add_argument("window-size=1920x1080")  # Set window size

    try:
        driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=chrome_options,
        )
    except Exception as e:
        pytest.fail(f"Failed to initialize WebDriver: {e}")

    yield driver  # Provide the driver to the test

    print("Tearing down Selenium WebDriver...")
    driver.quit()
    print("Selenium WebDriver torn down.")


def test_webdriver_initialization_only(browser):
    """
    Tests if the Selenium WebDriver initializes correctly.
    The Streamlit server part is disabled for this test.
    """
    print("Attempting to initialize WebDriver (browser fixture)...")
    assert browser is not None, "Browser fixture did not return a WebDriver instance."
    print(f"WebDriver initialized: {type(browser)}")

    # Optionally, try a very simple browser action that doesn't need a live server
    try:
        browser.get("data:,")  # Load a blank page
        print(f"Current URL after loading blank page: {browser.current_url}")
        assert (
            "data:," in browser.current_url
        ), "Browser did not navigate to blank page."
        print("WebDriver successfully loaded a blank page.")
    except Exception as e:
        pytest.fail(
            f"WebDriver failed during simple operation (loading blank page): {e}"
        )


if __name__ == "__main__":
    # For manual execution of this specific test file
    # Note: You might need to handle ChromeDriver path manually if not using pytest fixtures here.
    # Example: pytest tests/test_ui.py
    pytest.main([__file__, "-s", "-v"])  # -s for stdout, -v for verbose
