pre_emphasis = 0.97
emphasize_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

# FRAMING
frame_size = 0.025
frame_stride = 0.01
frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
signal_lenght = len(emphasize_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frame = int(numpy.ceil(float(numpy.abs(signal_lenght - frame_length)) / frame_step))

pad_signal_length = num_frame * frame_step + frame_length
z = numpy.zeros((pad_signal_length - signal_lenght))
pad_signal = numpy.append(emphasize_signal, z)

indices = numpy.tile(numpy.arange(0, frame_length), (num_frame, 1)) + numpy.tile(numpy.arange(0, num_frame * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(numpy.int32, copy=False)]

# FOURIER-TRANFORM AND POWER SPECTRUM
NFFT = 512
mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

# FILTER BANKS
nfilt = 40
low_freq_mel = 0
high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) /
700))
 # Convert Hz to Mel
mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt +
2)
 # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))
 # Convert Mel
to Hz
bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)
fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
f_m_minus = int(bin[m - 1])
 # left
f_m = int(bin[m])
 # center
f_m_plus = int(bin[m + 1])
 # right
for k in range(f_m_minus, f_m):
fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
for k in range(f_m, f_m_plus):
fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = numpy.dot(pow_frames, fbank.T)
filter_banks = numpy.where(filter_banks == 0,
numpy.finfo(float).eps, filter_banks)
 # Numerical Stability
filter_banks = 20 * numpy.log10(filter_banks)
 # dB
def preprocessing(path):
    y, sr = librosa.load(path, sr=22050)
    zero_padding = tf.zeros([111000] - tf.shape(y))
    y = tf.concat([y, zero_padding], 0)
    y = np.array(y)
    return y, sr

def add_gaussian_noise(y, noise_level=0.005):
    noise = noise_level * np.random.randn(len(y))
    y_noisy = y + noise
    return y_noisy

y, sr = librosa.load(path, sr=22050)

def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.expand_dims(mfcc, axis=2)
    return mfcc

def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = np.expand_dims(mfcc, axis=2)
    return mfcc

def add_noise(y, noise_level=0.005):
    noise = noise_level * np.random.randn(len(y))
    y_noisy = y + noise
    return y_noisy

def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = np.expand_dims(mfcc, axis=2)
    return mfcc

X_train, X_temp, y_train, y_temp =
train_test_split(X_features, y_labels, test_size=0.3,
random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp,
y_temp, test_size=0.5, random_state=42)

# model (a)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3, 3),
padding='same', activation='relu',
input_shape=X_train.shape[1:]))
model.add(tf.keras.layers.Conv2D(16, (3, 3),
padding='valid', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,
2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3),
padding='valid', activation='relu'))

model.add(tf.keras.layers.Conv2D(32, (3, 3),
padding='valid', activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3),
padding='valid', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,
2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4096,
activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(4096,
activation='relu'))
model.add(tf.keras.layers.Dense(5,
activation='softmax'))

#model (b)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(8, (3, 3),
padding='same', activation='relu',
input_shape=X_train.shape[1:]))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,
2)))
model.add(tf.keras.layers.Conv2D(16, (3, 3),
padding='valid', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,
2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3),
padding='valid', activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(500,
activation='relu'))
model.add(tf.keras.layers.Dense(450,
activation='relu'))
model.add(tf.keras.layers.Dense(5,
activation='softmax'))

model.fit(X_train, y_train, epochs=10, batch_size=32,
validation_data=(X_val, y_val))

def generate_features_and_labels():
    mfcc_features = []
    mfcc_labels = []
    folder_labels = []
    folder_path = glob.glob("")
    for folder in folder_path:
        folder_labels.append(folder[15:])
