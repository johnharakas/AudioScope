"""
This code creates a UDP server and listens for incoming datagrams from an android device microphone.
The microphone data is plotted in both the time and frequency domains.

It is intended to be used with this AudioLAN Android app. See: https://github.com/johnharakas/AudioLAN
"""
import select
import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import EngFormatter
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
from scipy.signal import butter, filtfilt
import socket
import struct
from multiprocessing import Process, Manager

FS = 48000
NYQUIST = 0.5 * FS
T = 1 / 10
ORDER = 6
CUTOFF = 2000

HOST = '192.168.1.165'
PORT = 8200

# Get buffer size based on sample rate (for android devices)
rate2buffer_map = {8000: 640,
                11025: 896,
                16000: 1280,
                22050: 1792,
                44100: 3584,
                48000: 3840,
                88200: 7104,
                96000: 7680,
                192000: 15360
                }

# Get sample rate based on packet size
buffer2rate_map = {u: v for v, u in rate2buffer_map.items()}

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.bind((HOST, PORT))


class Oscilloscope:
    def __init__(self, queue, buffer_size, sample_rate):
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate

        self.channels = [1]
        self.downsample = 1
        self.window = 640
        self.interval = 10

        self.fig, self.ax = plt.subplots(2)
        self.lines = []
        self.flines, = self.ax[1].plot([], [])

        self.chunk = int(self.window * self.sample_rate / (1000 * self.downsample))
        self.freq = np.ones(int(self.sample_rate) // 2)
        self.signal = np.zeros((self.chunk, len(self.channels)))

        self.yf = scipy.fft.fft(self.freq[-self.chunk:])
        self.yf = 2.0 / self.chunk * np.abs(self.yf[:self.chunk // 2])
        self.xf = np.linspace(0.0, 1.0 / (2.0 * (1 / self.sample_rate)), self.chunk // 2)

        self.start_stream(queue)

    def anim_fft(self, frame):
        yf = scipy.fft.fft(self.freq[-self.chunk:])
        yf = 2 / self.chunk * np.abs(yf[:self.chunk // 2])
        self.flines.set_data(self.xf, yf)
        return self.flines,

    def anim_signal(self, frame):
        for column, line in enumerate(self.lines):
            line.set_ydata(self.signal[:, column])
        return self.lines,

    def update(self, frame, queue):
        while True:
            if queue.empty():
                data = np.zeros((1, 1))
                break
            else:
                data = queue.get()
            shift = len(data)

            self.signal = np.roll(self.signal, -shift, axis=0)
            self.signal[-shift:, 0] = data
            self.freq = np.append(self.freq, data)

        self.freq = self.freq[-int(self.sample_rate) // 2:]

        signal_plot = self.anim_signal(frame)
        fft_plot = self.anim_fft(frame)
        return signal_plot + fft_plot

    def start_stream(self, queue):
        try:
            self.lines = self.ax[0].plot(self.signal)
            self.ax[0].axis((0, len(self.signal), -5000, 5000))
            self.ax[0].set(xlabel='Time', ylabel='Volume')

            self.ax[1].axis((0, self.sample_rate / 2, 0, 10))

            hz_format = EngFormatter(unit='Hz')
            self.ax[1].xaxis.set_major_formatter(hz_format)
            self.ax[1].set(xlabel='Frequency', ylabel='Amplitude')

            self.fig.tight_layout()
            print('Starting animation')
            anim = FuncAnimation(
                self.fig,
                self.update,
                interval=self.interval,
                fargs=(queue, ),
                blit=False
            )
            print('plotting')
            plt.show()

        except Exception as e:
            sys.exit(type(e).__name__ + ': ' + str(e))


def butter_lowpass_filter(data, cutoff=CUTOFF, fs=FS, order=ORDER):
    normal_cutoff = cutoff / NYQUIST
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def listen(socket_queue):
    while True:
        try:
            data, addr = udp.recvfrom(15360)
            try:
                dat = struct.unpack("%ih" % (len(data) // 2), data)
                socket_queue.put_nowait(dat)
            except OSError as e:
                print(e)
                break
        except (socket.timeout, KeyboardInterrupt) as e:
            print('Stopping')
            print(e)
            break

    print('Closing socket')
    udp.close()
    return


if __name__ == '__main__':

    manager = Manager()
    queue = manager.Queue()
    buffer_size = 15360
    sample_rate = 0
    timeout = 120

    print('Listening for UDP packets on port %d' % PORT)

    # Listen for first packets, timeout if none received
    ready = select.select([udp], [], [], timeout)

    if ready[0]:
        data, addr = udp.recvfrom(buffer_size)
        # Set timeout for 120 seconds of inactivity
        udp.settimeout(timeout)
        if data:
            print('Receiving packet of size {} from {}'.format(len(data), addr))
            buffer_size = len(data)
            sample_rate = buffer2rate_map[buffer_size]
            print('Determining sample rate is %dHz' % sample_rate)
        try:
            p = Process(target=Oscilloscope, args=(queue, buffer_size, sample_rate))
            p.start()
            listen(queue)
            p.join()
        except KeyboardInterrupt as k:
            print(k)
    else:
        print('Timeout: nothing received after %d seconds.' % timeout)
        udp.close()
