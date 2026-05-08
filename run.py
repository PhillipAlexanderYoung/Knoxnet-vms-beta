import subprocess
import time
import sys
import os
import signal
import threading
import requests
from requests.auth import HTTPBasicAuth
from pathlib import Path
from datetime import datetime
import platform
import shutil
import tarfile
import zipfile
import io


class ServerManager:
    def __init__(self):
        self.mediamtx_process = None
        self.flask_process = None
        self.running = True
        self.mediamtx_available = False
        self.mediamtx_restart_count = 0
        self.mediamtx_max_restarts = 5
        self.mediamtx_is_critical = True  # CRITICAL SERVICE FLAG

    def _platform_target(self):
        system = platform.system().lower()
        machine = platform.machine().lower()

        if machine in {"x86_64", "amd64"}:
            arch = "amd64"
        elif machine in {"aarch64", "arm64"}:
            arch = "arm64"
        else:
            arch = None

        if system.startswith("win"):
            os_id = "windows"
        elif system == "darwin":
            os_id = "darwin"
        elif system == "linux":
            os_id = "linux"
        else:
            os_id = None

        return os_id, arch

    def _mediamtx_release_url(self):
        # Allow overrides for enterprise packaging or air-gapped installs
        override_url = os.environ.get("MEDIAMTX_DOWNLOAD_URL", "").strip()
        if override_url:
            return override_url

        version = os.environ.get("MEDIAMTX_VERSION", "v1.10.0").strip()
        os_id, arch = self._platform_target()
        if not os_id or not arch:
            return None

        if os_id == "windows":
            filename = f"mediamtx_{version}_{os_id}_{arch}.zip"
        else:
            filename = f"mediamtx_{version}_{os_id}_{arch}.tar.gz"

        return f"https://github.com/bluenviron/mediamtx/releases/download/{version}/{filename}"

    def _download_and_install_mediamtx(self, target_dir: Path):
        url = self._mediamtx_release_url()
        if not url:
            print("ERROR: Unsupported platform for automatic MediaMTX download")
            return False

        print(f"Downloading MediaMTX from: {url}")
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            data = resp.content
        except Exception as e:
            print(f"ERROR: Failed to download MediaMTX: {e}")
            return False

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    zf.extractall(target_dir)
            else:
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                    tf.extractall(target_dir)

            # Find extracted binary (platform-aware)
            os_id, _arch = self._platform_target()
            if os_id == "windows":
                candidates = [
                    target_dir / "mediamtx.exe",
                    target_dir / "mediamtx",
                ]
            else:
                candidates = [
                    target_dir / "mediamtx",
                    target_dir / "mediamtx.exe",
                ]

            for c in candidates:
                if c.exists() and c.is_file():
                    if c.suffix != ".exe":
                        c.chmod(c.stat().st_mode | 0o111)
                    self.mediamtx_path = c
                    self.mediamtx_available = True
                    print(f"MediaMTX installed to: {c}")
                    return True

            print("ERROR: MediaMTX binary not found after extraction")
            return False
        except Exception as e:
            print(f"ERROR: Failed to install MediaMTX: {e}")
            return False

    def check_mediamtx_availability(self):
        """Check if MediaMTX executable exists or install it automatically."""
        os_id, _arch = self._platform_target()
        possible_paths = []

        # Also check the frozen bundle extraction dir (PyInstaller onefile) and the executable directory.
        # This keeps portable builds working even when the process CWD isn't the app folder.
        base_dirs = []
        try:
            base_dirs.append(Path("."))
        except Exception:
            pass
        try:
            base_dirs.append(Path(sys.executable).resolve().parent)
        except Exception:
            pass
        try:
            meipass = getattr(sys, "_MEIPASS", "")  # PyInstaller onefile extraction dir
            if meipass:
                base_dirs.append(Path(str(meipass)))
        except Exception:
            pass

        if os_id == "windows":
            for b in base_dirs:
                possible_paths.extend([
                    b / "mediamtx" / "mediamtx.exe",
                    b / "mediamtx.exe",
                ])
        else:
            for b in base_dirs:
                possible_paths.extend([
                    b / "mediamtx" / "mediamtx",
                    b / "mediamtx",
                ])

        # Check repo paths (platform-appropriate)
        for path in possible_paths:
            if path.exists() and path.is_file():
                self.mediamtx_path = path
                self.mediamtx_available = True
                return True

        # Check PATH
        system_path = shutil.which("mediamtx")
        if system_path:
            self.mediamtx_path = Path(system_path)
            self.mediamtx_available = True
            return True

        # Auto-download into ./mediamtx
        return self._download_and_install_mediamtx(Path("./mediamtx"))

    def check_mediamtx_health(self):
        """Health check for MediaMTX API"""
        try:
            # Use authentication only when explicitly configured.
            username = (os.environ.get("MEDIAMTX_API_USERNAME") or "").strip()
            password = (os.environ.get("MEDIAMTX_API_PASSWORD") or "").strip()
            auth = HTTPBasicAuth(username, password) if username and password else None

            response = requests.get(
                "http://localhost:9997/v3/config/global/get",
                timeout=2,
                auth=auth,
            )
            # 200 = authorized. 401/403 = API is up but auth is required.
            return response.status_code in {200, 401, 403}
        except Exception:
            return False

    def restart_mediamtx(self):
        """Restart MediaMTX server (CRITICAL SERVICE)"""
        if self.mediamtx_restart_count >= self.mediamtx_max_restarts:
            print(f"CRITICAL ERROR: MediaMTX reached max restart limit ({self.mediamtx_max_restarts})")
            return False

        self.mediamtx_restart_count += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\nRESTART: [{timestamp}] RESTARTING MediaMTX (Attempt {self.mediamtx_restart_count}/{self.mediamtx_max_restarts})...")

        # Stop existing process
        if self.mediamtx_process:
            try:
                self.mediamtx_process.terminate()
                self.mediamtx_process.wait(timeout=5)
            except:
                try:
                    self.mediamtx_process.kill()
                except:
                    pass

        # Wait before restart
        time.sleep(2)

        # Restart
        return self.start_mediamtx()

    def trigger_camera_reconnection(self):
        """Trigger backend to reconnect all cameras to MediaMTX"""
        try:
            print("   Triggering camera reconnection...")
            response = requests.post(
                "http://localhost:5000/api/cameras/connect",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                print(f"   Reconnected {data.get('connected_count', 0)} cameras")
            else:
                print(f"   WARNING: Camera reconnection returned {response.status_code}")
        except Exception as e:
            print(f"   WARNING: Failed to trigger camera reconnection: {e}")

    def start_mediamtx(self):
        """Start MediaMTX server if available - CRITICAL SERVICE"""
        if not self.mediamtx_available:
            if self.mediamtx_is_critical:
                print("CRITICAL ERROR: MediaMTX not found - camera streaming requires MediaMTX!")
                print("   Download from: https://github.com/bluenviron/mediamtx/releases")
                print("   Extract to ./mediamtx/ directory")
                return False  # Don't continue without critical service
            else:
                print("WARNING: MediaMTX not found - starting without streaming server")
                return True

        # If MediaMTX is already running (common on Windows when started elsewhere),
        # don't fail the whole stack; just reuse it.
        try:
            if self.check_mediamtx_health():
                print("MediaMTX already running - skipping start")
                return True
        except Exception:
            pass

        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] Starting MediaMTX server (CRITICAL SERVICE)...")
            print(f"   Using: {self.mediamtx_path}")
            print(f"   Config: {self.mediamtx_path.parent / 'mediamtx.yml'}")

            # Start MediaMTX process with explicit config to avoid wrong file being loaded
            config_path = (
                self.mediamtx_path.parent / 'mediamtx.yml'
                if self.mediamtx_path.parent.name == 'mediamtx'
                else Path('mediamtx.yml')
            )

            exe_path = self.mediamtx_path.resolve()
            self.mediamtx_process = subprocess.Popen(
                # MediaMTX uses environment variable MTX_CONFIG for config path, or no args for default
                # But for local execution, we don't need -config flag, we just point env var or run in dir
                # If we must specify config, some versions use just the path as argument
                # Let's try setting the environment variable instead as it's safer across versions
                [str(exe_path)],
                env={**os.environ, 'MTX_CONFIG_PATH': str(config_path)},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.mediamtx_path.parent if self.mediamtx_path.parent.name == "mediamtx" else "."
            )

            # Give MediaMTX time to start
            print("   Waiting for MediaMTX to initialize...")
            time.sleep(3)

            if self.mediamtx_process.poll() is None:
                # Verify API is responding
                if self.check_mediamtx_health():
                    print("MediaMTX server started successfully")
                    print("   WebRTC: http://localhost:8889")
                    print("   HLS: http://localhost:8888")
                    print("   API: http://localhost:9997")
                    print("   RTSP: rtsp://localhost:8554")
                    return True
                else:
                    print("WARNING: MediaMTX started but API not responding yet...")
                    time.sleep(2)
                    if self.check_mediamtx_health():
                        print("MediaMTX API now responding")
                        return True
                    else:
                        print("ERROR: MediaMTX API failed to respond")
                        return False if self.mediamtx_is_critical else True
            else:
                print("ERROR: MediaMTX failed to start")
                stdout, stderr = self.mediamtx_process.communicate()
                if stdout:
                    print(f"   Output: {stdout}")
                if stderr:
                    print(f"   Error: {stderr}")
                return False if self.mediamtx_is_critical else True

        except Exception as e:
            print(f"ERROR starting MediaMTX: {e}")
            return False if self.mediamtx_is_critical else True

    def start_flask(self):
        """Start Flask application"""
        try:
            print("Starting Flask application...")

            # Check if app.py exists
            if not Path("app.py").exists():
                print("ERROR: app.py not found in current directory")
                return False

            self.flask_process = subprocess.Popen(
                [sys.executable, "app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Monitor Flask output in a separate thread
            def monitor_flask():
                for line in iter(self.flask_process.stdout.readline, ''):
                    if line.strip():
                        # Filter out standard werkzeug request logs to reduce noise, but keep errors/info
                        line_str = line.strip()
                        if " - - [" not in line_str and "GET /health" not in line_str:
                             print(f"[Flask] {line_str}")
                        # Also print the specific network diagnostics we are looking for
                        if "NETWORK DIAGNOSTICS" in line_str or "DNS Resolve" in line_str:
                             print(f"[Flask] {line_str}")

            threading.Thread(target=monitor_flask, daemon=True).start()

            # Wait a bit for Flask to start
            print("   Waiting for Flask to initialize...")
            time.sleep(5)

            if self.flask_process.poll() is None:
                print("Flask application started successfully")
                print("   API available at: http://localhost:5000/api")
                print("   Web interface: http://localhost:5000")
                return True
            else:
                print("ERROR: Flask failed to start")
                return False

        except Exception as e:
            print(f"ERROR starting Flask: {e}")
            return False

    def stop_servers(self):
        """Stop both servers gracefully and ensure no child processes remain"""
        print("\nStopping servers...")
        self.running = False

        def kill_process_tree(process):
            if not process:
                return
            
            pid = process.pid
            try:
                # Attempt to use psutil for clean tree killing if available
                import psutil
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.terminate()
                parent.terminate()
                
                # Wait briefly then force kill if still alive
                gone, alive = psutil.wait_procs(children + [parent], timeout=3)
                for p in alive:
                    p.kill()
            except ImportError:
                # Fallback to Windows taskkill for thorough tree cleanup
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], 
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
            except Exception:
                # Last resort simple kill
                try:
                    process.kill()
                except:
                    pass

        if self.flask_process:
            print("   Stopping Flask and children...")
            kill_process_tree(self.flask_process)
            print("   Flask stopped")

        if self.mediamtx_process:
            print("   Stopping MediaMTX and children...")
            kill_process_tree(self.mediamtx_process)
            print("   MediaMTX stopped")

        print("All servers stopped")

    def run(self):
        """Start both servers and wait"""

        def signal_handler(signum, frame):
            self.stop_servers()
            sys.exit(0)

        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)

        print("=" * 50)
        print("Knoxnet VMS Beta Server Manager")
        print("=" * 50)

        # Check MediaMTX availability
        self.check_mediamtx_availability()

        # Start MediaMTX first (or skip if not available)
        if not self.start_mediamtx():
            print("ERROR: Failed to start servers.")
            return

        # Start Flask
        if not self.start_flask():
            print("ERROR: Failed to start Flask.")
            self.stop_servers()
            return

        print("\n" + "=" * 50)
        if self.mediamtx_available:
            print("BOTH SERVERS ARE RUNNING!")
            print("=" * 50)
            print("MediaMTX WebRTC: http://localhost:8889")
            print("MediaMTX API: http://localhost:9997")
        else:
            print("FLASK SERVER IS RUNNING!")
            print("=" * 50)
            print("WARNING: MediaMTX not available - limited streaming features")

        print("Flask API: http://localhost:5000/api")
        print("Web Interface: http://localhost:5000")
        print("=" * 50)
        print("Press Ctrl+C to stop servers")
        print("=" * 50)

        # Keep running until interrupted with health monitoring
        last_health_check = time.time()
        health_check_interval = 30  # Check every 30 seconds
        
        try:
            while self.running:
                # Check if processes are still alive
                if self.mediamtx_process and self.mediamtx_process.poll() is not None:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"\nERROR: [{timestamp}] CRITICAL: MediaMTX stopped unexpectedly")
                    if self.mediamtx_is_critical:
                        print("   Attempting automatic restart...")
                        if self.restart_mediamtx():
                            print("   MediaMTX restarted successfully")
                            # Trigger camera reconnection
                            self.trigger_camera_reconnection()
                        else:
                            print("   Failed to restart MediaMTX - shutting down")
                            break
                    else:
                        print("   Continuing without MediaMTX")
                        
                if self.flask_process and self.flask_process.poll() is not None:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"\nERROR: [{timestamp}] Flask stopped unexpectedly")
                    break
                
                # Periodic health checks for MediaMTX
                current_time = time.time()
                if current_time - last_health_check >= health_check_interval:
                    if self.mediamtx_process and self.mediamtx_process.poll() is None:
                        if not self.check_mediamtx_health():
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            print(f"\nWARNING: [{timestamp}] MediaMTX health check failed")
                            if self.mediamtx_is_critical:
                                print("   Attempting restart...")
                                if self.restart_mediamtx():
                                    print("   MediaMTX recovered")
                                    self.trigger_camera_reconnection()
                                else:
                                    print("   MediaMTX recovery failed")
                    last_health_check = current_time
                    
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nReceived shutdown signal...")
        finally:
            self.stop_servers()


if __name__ == "__main__":
    manager = ServerManager()
    manager.run()