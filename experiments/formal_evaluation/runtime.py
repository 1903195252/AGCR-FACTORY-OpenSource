"""Bounded inference jobs stop between complete atomic clip/episode outputs."""
import signal,time

class Budget:
    def __init__(self,gpu_hours):
        if gpu_hours<=0:raise ValueError('Positive explicit inference GPU-hour allowance required')
        self.started=time.monotonic();self.hours=float(gpu_hours);self.stop=False
        signal.signal(signal.SIGTERM,self._stop);signal.signal(signal.SIGINT,self._stop)
    def _stop(self,*unused):self.stop=True
    def expired(self):return self.stop or self.elapsed_hours()>=self.hours
    def elapsed_hours(self):return (time.monotonic()-self.started)/3600
