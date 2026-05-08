import os
import socket, time, keyboard

IP   = os.environ.get("GATE_CONTROLLER_IP", "0.0.0.0")     # set via env or config
PORT = int(os.environ.get("GATE_CONTROLLER_PORT", "7777"))
SECRET = os.environ.get("GATE_CONTROLLER_SECRET", "")
RAMP_TIME = 0.8            # seconds for full throttle
STEP_MS = 0.05             # step delay

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
level = 0.0                # -1..+1 throttle value
target = 0.0

def send(cmd):
    msg = f"KEY {SECRET} {cmd}"
    sock.sendto(msg.encode(), (IP, PORT))
    print(f"> {cmd}")

def smooth_ramp(target_value):
    global level
    step = (target_value - level) / (RAMP_TIME / STEP_MS)
    for _ in range(int(RAMP_TIME / STEP_MS)):
        level += step
        msg = f"KEY {SECRET} SPD {level:.2f}"
        sock.sendto(msg.encode(), (IP, PORT))
        time.sleep(STEP_MS)
    level = target_value

print("Use ↑ for OPEN, ↓ for CLOSE, SPACE to STOP, ESC to exit.")
try:
    while True:
        if keyboard.is_pressed("up"):
            if target != 1.0:
                target = 1.0
                smooth_ramp(target)
                send("OPEN")
        elif keyboard.is_pressed("down"):
            if target != -1.0:
                target = -1.0
                smooth_ramp(target)
                send("CLOSE")
        elif keyboard.is_pressed("space"):
            target = 0.0
            smooth_ramp(target)
            send("STOP")
        elif keyboard.is_pressed("esc"):
            break
        time.sleep(0.05)
finally:
    smooth_ramp(0.0)
    send("STOP")
    sock.close()
    print("Exited cleanly.")
