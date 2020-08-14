"""
The essence of this code came from the link below. I just encapsulated it into a class and added a frequency plot
as well. This will be used alongside Qt5.
https://github.com/spatialaudio/python-sounddevice/blob/40e6380f93456e3843798eced3189da8780ba092/examples/plot_input.py
"""

import queue
import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import EngFormatter
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import scipy.fftpack


class Oscilloscope:
    def __init__(self, device='default'):
        self.queue = queue.Queue()
        self.devices = sd.query_devices()
        self.devices = list(filter(lambda d: d['max_input_channels'] > 0, self.devices))  # Limit to mics
        self.default = sd.query_devices(device)
        self.sample_rate = int(self.default['default_samplerate'])
        self.channels = [1]
        self.cmap = [c - 1 for c in self.channels]
        self.downsample = 1
        self.window = 200
        self.interval = 30

        self.fig, self.ax = plt.subplots(2)
        self.lines = []
        self.flines, = self.ax[1].plot([], [])

        self.chunk = int(self.window * self.sample_rate / (1000 * self.downsample))
        self.freq = np.ones(int(self.sample_rate) // 2)
        self.signal = np.zeros((self.chunk, len(self.channels)))

        self.yf = scipy.fft.fft(self.freq[-self.chunk:])
        self.yf = 2.0 / self.chunk * np.abs(self.yf[:self.chunk // 2])
        self.xf = np.linspace(0.0, 1.0 / (2.0 * (1 / self.sample_rate)), self.chunk // 2)

    def anim_fft(self, frame):
        yf = scipy.fft.fft(self.freq[-self.chunk:])
        yf = 2 / self.chunk * np.abs(yf[:self.chunk // 2])
        self.flines.set_data(self.xf, yf)
        return self.flines,

    def anim_signal(self, frame):
        for column, line in enumerate(self.lines):
            line.set_ydata(self.signal[:, column])
        return self.lines,

    def update(self, frame):
        while True:
            try:
                data = self.queue.get_nowait()
            except queue.Empty as e:
                break
            shift = len(data)

            self.signal = np.roll(self.signal, -shift, axis=0)
            self.signal[-shift:, :] = data
            self.freq = np.append(self.freq, data)

        self.freq = self.freq[-int(self.sample_rate) // 2:]

        signal_plot = self.anim_signal(frame)
        fft_plot = self.anim_fft(frame)
        return signal_plot + fft_plot

    def callback(self, data, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.queue.put(data[::self.downsample, self.cmap])

    def start(self):
        try:
            self.lines = self.ax[0].plot(self.signal)
            self.ax[0].axis((0, len(self.signal), -1, 1))
            self.ax[0].set(xlabel='Time', ylabel='Volume')

            self.ax[1].axis((0, self.sample_rate / 2, 0, 0.05))
            hz_format = EngFormatter(unit='Hz')
            # self.ax[1].set_xticks([1, 4, 5])
            self.ax[1].xaxis.set_major_formatter(hz_format)
            self.ax[1].set(xlabel='Frequency', ylabel='Amplitude')

            self.fig.tight_layout()
            print('Opening stream...')
            print('device: {}'.format(self.default['name']))
            print('channels: {}'.format(self.channels))
            print('sample rate: {}'.format(self.sample_rate))

            stream = sd.InputStream(
                device=self.default['name'],
                channels=max(self.channels),
                samplerate=self.sample_rate,
                callback=self.callback
            )
            print('Starting animation')
            anim = FuncAnimation(
                self.fig,
                self.update,
                interval=self.interval,
                blit=False
            )
            with stream:
                print('plotting')
                plt.show()
        except Exception as e:
            sys.exit(type(e).__name__ + ': ' + str(e))


scope = Oscilloscope(device='USB Device 0x46d:0x825')
scope.start()
