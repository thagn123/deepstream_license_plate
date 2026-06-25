import psutil, os, signal
for p in psutil.process_iter(['pid', 'cmdline']):
    if p.info['cmdline'] and 'kafka_consumer.py' in ' '.join(p.info['cmdline']):
        print(f"Killing {p.pid}")
        try: os.kill(p.pid, signal.SIGKILL)
        except Exception: pass
