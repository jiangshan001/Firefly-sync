from gpiozero import LED
from time import sleep, time
from pathlib import Path
import csv

LED_PIN = 17
ON_TIME = 0.2
OFF_TIME = 0.8
CYCLES = 10

led = LED(LED_PIN)

log_dir = Path.home() / "firefly_drone" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "led_blink_test.csv"

print(f"Blinking LED on GPIO{LED_PIN}")
print(f"Log file: {log_file}")

try:
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "state"])

        for i in range(CYCLES):
            led.on()
            t_on = time()
            writer.writerow([t_on, "ON"])
            print(f"{i+1:02d}: LED ON")
            sleep(ON_TIME)

            led.off()
            t_off = time()
            writer.writerow([t_off, "OFF"])
            print(f"{i+1:02d}: LED OFF")
            sleep(OFF_TIME)

finally:
    led.off()
    print("Done. LED OFF.")
