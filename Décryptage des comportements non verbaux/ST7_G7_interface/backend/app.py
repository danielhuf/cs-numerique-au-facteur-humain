from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json

from torch.utils.data import Dataset
import numpy as np
import torch
import resampy
from transformers import BertModel, AutoTokenizer, get_linear_schedule_with_warmup
from huggingsound import SpeechRecognitionModel
from scipy.io.wavfile import read
from torch import nn
from collections import OrderedDict
from pydub import AudioSegment
import os
import soundfile as sf 
import librosa
import glob
import shutil


modelSpeechRecognition = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
device = torch.device("cuda")
print(device)

# Hyperparameters used in feature and example generation.
NUM_FRAMES = 96*2
NUM_MELS = 64*2
EXAMPLE_SIZE = 3
READ_OR_GEN = True # True -> Read the data. False -> Generate the data (Around one hour). If example_size!=3, put READ_OR_GEN = False


# Hyperparameters used in training.
LEARNING_RATE = 2e-5 # Learning rate for the Adam optimizer.
BATCH_SIZE = 32
NUM_EPOCHS = 25
OPTIONS = 2
SEED = 71

EMOTIONS = ['neu', 'fru', 'ang', 'sad', 'exc'] 

# Configurating some parameters
torch.manual_seed(SEED)
np.random.seed(SEED)
EMOTIONS = {emo: i for i, emo in enumerate(EMOTIONS)}
EMBEDDING_SIZE = len(EMOTIONS)  # Size of embedding layer.

class Preprocess_audio(Dataset):
    def __init__(self, data: np.array):
        super().__init__()
        # Mel spectrum constants and functions.
        self.data = data
        self.MEL_BREAK_FREQUENCY_HERTZ = 700.0
        self.MEL_HIGH_FREQUENCY_Q = 1127.0
        self.MEL_MIN_HZ = 725
        self.MEL_MAX_HZ = 7500
        self.EXAMPLE_SIZE = EXAMPLE_SIZE # seconds
        self.EXAMPLE_WINDOW_SECONDS = 0.96*self.EXAMPLE_SIZE
        self.EXAMPLE_HOP_SECONDS = 0.96
        self.SAMPLE_RATE = 16000

        self.NUM_FRAMES = NUM_FRAMES/self.EXAMPLE_SIZE  # Frames in input mel-spectrogram patch for each second
        self.NUM_BANDS = NUM_MELS  # Frequency bands in input mel-spectrogram patch.
        self.STFT_WINDOW_LENGTH_SECONDS = 0.025
        self.STFT_HOP_LENGTH_SECONDS = 0.010*96/self.NUM_FRAMES
        self.NUM_MEL_BINS = self.NUM_BANDS


    def frame(self, data, window_length, hop_length):
      """Convert array into a sequence of successive possibly overlapping frames.
      An n-dimensional array of shape (num_samples, ...) is converted into an
      (n+1)-D array of shape (num_frames, window_length, ...), where each frame
      starts hop_length points after the preceding one.
      This is accomplished using stride_tricks, so the original data is not
      copied.  However, there is no zero-padding, so any incomplete frames at the
      end are not included.
      Args:
        data: np.array of dimension N >= 1.
        window_length: Number of samples in each frame.
        hop_length: Advance (in samples) between each window.
      Returns:
        (N+1)-D np.array with as many rows as there are complete frames that can be
        extracted.
      """
      num_samples = data.shape[0]
      num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
      shape = (num_frames, window_length) + data.shape[1:]
      strides = (data.strides[0] * hop_length,) + data.strides
      return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


    def periodic_hann(self, window_length):
      """Calculate a "periodic" Hann window.
      The classic Hann window is defined as a raised cosine that starts and
      ends on zero, and where every value appears twice, except the middle
      point for an odd-length window.  Matlab calls this a "symmetric" window
      and np.hanning() returns it.  However, for Fourier analysis, this
      actually represents just over one cycle of a period N-1 cosine, and
      thus is not compactly expressed on a length-N Fourier basis.  Instead,
      it's better to use a raised cosine that ends just before the final
      zero value - i.e. a complete cycle of a period-N cosine.  Matlab
      calls this a "periodic" window. This routine calculates it.
      Args:
        window_length: The number of points in the returned window.
      Returns:
        A 1D np.array containing the periodic hann window.
      """
      return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
                                np.arange(window_length)))


    def stft_magnitude(self, signal, fft_length,
                      hop_length=None,
                      window_length=None):
      """Calculate the short-time Fourier transform magnitude.
      Args:
        signal: 1D np.array of the input time-domain signal.
        fft_length: Size of the FFT to apply.
        hop_length: Advance (in samples) between each frame passed to FFT.
        window_length: Length of each block of samples to pass to FFT.
      Returns:
        2D np.array where each row contains the magnitudes of the fft_length/2+1
        unique values of the FFT for the corresponding frame of input samples.
      """
      frames = self.frame(signal, window_length, hop_length)
      # Apply frame window to each frame. We use a periodic Hann (cosine of period
      # window_length) instead of the symmetric Hann of np.hanning (period
      # window_length-1).
      window = self.periodic_hann(window_length)
      windowed_frames = frames * window
      return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))





    def hertz_to_mel(self, frequencies_hertz):
      """Convert frequencies to mel scale using HTK formula.
      Args:
        frequencies_hertz: Scalar or np.array of frequencies in hertz.
      Returns:
        Object of same size as frequencies_hertz containing corresponding values
        on the mel scale.
      """
      return self.MEL_HIGH_FREQUENCY_Q * np.log(
          1.0 + (frequencies_hertz / self.MEL_BREAK_FREQUENCY_HERTZ))


    def spectrogram_to_mel_matrix(self, num_mel_bins=20,
                                  num_spectrogram_bins=129,
                                  audio_sample_rate=8000,
                                  lower_edge_hertz=125.0,
                                  upper_edge_hertz=3800.0):
      """Return a matrix that can post-multiply spectrogram rows to make mel.
      Returns a np.array matrix A that can be used to post-multiply a matrix S of
      spectrogram values (STFT magnitudes) arranged as frames x bins to generate a
      "mel spectrogram" M of frames x num_mel_bins.  M = S A.
      The classic HTK algorithm exploits the complementarity of adjacent mel bands
      to multiply each FFT bin by only one mel weight, then add it, with positive
      and negative signs, to the two adjacent mel bands to which that bin
      contributes.  Here, by expressing this operation as a matrix multiply, we go
      from num_fft multiplies per frame (plus around 2*num_fft adds) to around
      num_fft^2 multiplies and adds.  However, because these are all presumably
      accomplished in a single call to np.dot(), it's not clear which approach is
      faster in Python.  The matrix multiplication has the attraction of being more
      general and flexible, and much easier to read.
      Args:
        num_mel_bins: How many bands in the resulting mel spectrum.  This is
          the number of columns in the output matrix.
        num_spectrogram_bins: How many bins there are in the source spectrogram
          data, which is understood to be fft_size/2 + 1, i.e. the spectrogram
          only contains the nonredundant FFT bins.
        audio_sample_rate: Samples per second of the audio at the input to the
          spectrogram. We need this to figure out the actual frequencies for
          each spectrogram bin, which dictates how they are mapped into mel.
        lower_edge_hertz: Lower bound on the frequencies to be included in the mel
          spectrum.  This corresponds to the lower edge of the lowest triangular
          band.
        upper_edge_hertz: The desired top edge of the highest frequency band.
      Returns:
        An np.array with shape (num_spectrogram_bins, num_mel_bins).
      Raises:
        ValueError: if frequency edges are incorrectly ordered or out of range.
      """
      nyquist_hertz = audio_sample_rate / 2.
      if lower_edge_hertz < 0.0:
        raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
      if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                        (lower_edge_hertz, upper_edge_hertz))
      if upper_edge_hertz > nyquist_hertz:
        raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" %
                        (upper_edge_hertz, nyquist_hertz))
      spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
      spectrogram_bins_mel = self.hertz_to_mel(spectrogram_bins_hertz)
      # The i'th mel band (starting from i=1) has center frequency
      # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
      # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
      # the band_edges_mel arrays.
      band_edges_mel = np.linspace(self.hertz_to_mel(lower_edge_hertz),
                                  self.hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
      # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
      # of spectrogram values.
      mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
      for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
        # Calculate lower and upper slopes for every spectrogram bin.
        # Line segments are linear in the *mel* domain, not hertz.
        lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                      (center_mel - lower_edge_mel))
        upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                      (upper_edge_mel - center_mel))
        # .. then intersect them with each other and zero.
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                              upper_slope))
      # HTK excludes the spectrogram DC bin; make sure it always gets a zero
      # coefficient.
      mel_weights_matrix[0, :] = 0.0
      return mel_weights_matrix


    def log_mel_spectrogram(self, data,
                            audio_sample_rate=8000,
                            log_offset=0.0,
                            window_length_secs=0.025,
                            hop_length_secs=0.010,
                            **kwargs):
      """Convert waveform to a log magnitude mel-frequency spectrogram.
      Args:
        data: 1D np.array of waveform data.
        audio_sample_rate: The sampling rate of data.
        log_offset: Add this to values when taking log to avoid -Infs.
        window_length_secs: Duration of each window to analyze.
        hop_length_secs: Advance between successive analysis windows.
        **kwargs: Additional arguments to pass to spectrogram_to_mel_matrix.
      Returns:
        2D np.array of (num_frames, num_mel_bins) consisting of log mel filterbank
        magnitudes for successive frames.
      """
      window_length_samples = int(round(audio_sample_rate * window_length_secs))
      hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
      fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
      spectrogram = self.stft_magnitude(
          data,
          fft_length=fft_length,
          hop_length=hop_length_samples,
          window_length=window_length_samples)
      mel_spectrogram = np.dot(spectrogram, self.spectrogram_to_mel_matrix(
          num_spectrogram_bins=spectrogram.shape[1],
          audio_sample_rate=audio_sample_rate, **kwargs))
      return np.log(mel_spectrogram + log_offset)


    def make_chunks_of_n_seconds(self, wav_data, sr):
      size = wav_data.shape[0]
      # Convert to mono.
      if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)
      if size>=(sr*self.EXAMPLE_SIZE):
        data = wav_data[:sr*self.EXAMPLE_SIZE]
      else:
        data = np.concatenate((wav_data, np.zeros(sr*self.EXAMPLE_SIZE-size)))
      return data


    def wave_to_examples(self, data, sample_rate, return_tensor = False, gen_chunk = False):
        """Converts audio waveform into an array of examples for VGGish.
      Args:
        data: np.array of either one dimension (mono) or two dimensions
          (multi-channel, with the outer dimension representing channels).
          Each sample is generally expected to lie in the range [-1.0, +1.0],
          although this is not required.
        sample_rate: Sample rate of data.
        return_tensor: Return data as a Pytorch tensor ready for VGGish
      Returns:
        3-D np.array of shape [num_examples, num_frames, num_bands] which represents
        a sequence of examples, each of which contains a patch of log mel
        spectrogram, covering num_frames frames of audio and num_bands mel frequency
        bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
      """
        # Convert to mono.
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        # Resample to the rate assumed by VGGish.
        if sample_rate != self.SAMPLE_RATE:
            data = resampy.resample(data, sample_rate, self.SAMPLE_RATE)
        if gen_chunk:
              data = self.make_chunks_of_n_seconds(data, sample_rate)

        # Compute log mel spectrogram features.
        log_mel = self.log_mel_spectrogram(
            data,
            audio_sample_rate=self.SAMPLE_RATE,
            log_offset=0.01,
            window_length_secs=self.STFT_WINDOW_LENGTH_SECONDS,
            hop_length_secs=self.STFT_HOP_LENGTH_SECONDS,
            num_mel_bins=self.NUM_MEL_BINS,
            lower_edge_hertz=self.MEL_MIN_HZ,
            upper_edge_hertz=self.MEL_MAX_HZ)

        # Frame features into examples.
        features_sample_rate = 1.0 / self.STFT_HOP_LENGTH_SECONDS
        example_window_length = int(round(
            self.EXAMPLE_WINDOW_SECONDS * features_sample_rate))

        example_hop_length = int(round(
            self.EXAMPLE_HOP_SECONDS * features_sample_rate))

        log_mel_examples = self.frame(
            log_mel,
            window_length=example_window_length,
            hop_length=example_hop_length)

        if return_tensor:
            log_mel_examples = torch.tensor(
                log_mel_examples, requires_grad=True)[:, None, :, :].float()
            log_mel = torch.tensor(
                log_mel, requires_grad=True)[None, :, :].float()
            
        return log_mel_examples


    def __len__(self):
      return self.data.shape[0]

    def __getitem__(self, index: int):
      return self.wave_to_examples(self.data[index], self.SAMPLE_RATE, True)


class Preprocess_text(Dataset):

  def __init__(
    self, 
    data: np.array
    ):
    self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    self.data = data
    self.size = self.data.shape[0]
    self.max_token_len = 128
    
  def __len__(self):
    return self.size

  def __getitem__(self, index: int):
    phrases = self.data[index]
    encoding = self.tokenizer.encode_plus(
      phrases,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return encoding["input_ids"].flatten(), encoding["attention_mask"].flatten()


class Audio_model(torch.nn.Module):
    def __init__(self, EMBEDDING_SIZE = 5, pre_train = True, class_weights = None):
        super().__init__()
        model = torch.hub.load('harritaylor/torchvggish', model='vggish', pretrained = pre_train)

        vggish_pretrained_features = model.features
        for param in vggish_pretrained_features[:11].parameters():
              param.requires_grad = not pre_train
        for param in vggish_pretrained_features[11:].parameters():
              param.requires_grad = True
        self.features = vggish_pretrained_features

        vggish_embeddings = nn.Sequential(OrderedDict([('0', nn.Linear(32*NUM_MELS, 64)), ('1', nn.ReLU())]))

        classifier = nn.Sequential(OrderedDict([('0', nn.Linear(64, EMBEDDING_SIZE, bias = False)), ('1', nn.Softmax())]))
        self.classifier = classifier

        self.loss_function = nn.CrossEntropyLoss(weight=class_weights, reduce='mean')


    def to_embeddings(self, x):
        x = x.mean(axis=-2)
        x = x.flatten(-2,-1)
        x = self.embeddings(x)

        return x


    def forward(self, x):
        x = self.features(x)
        x = self.to_embeddings(x)
        x = self.classifier(x)

        return x


    def predict(self, x):
        x = self.forward(x)
        x = torch.argmax(x, dim=1)

        return x
    
class Text_model(torch.nn.Module):
    def __init__(self, EMBEDDING_SIZE = 5, class_weights = None):
        super().__init__()
        self.features = BertModel.from_pretrained("bert-base-cased", return_dict=True)
        self.embeddings = nn.Sequential(OrderedDict([('0', nn.Linear(self.features.config.hidden_size, 64)), ('1', nn.ReLU())]))
        self.classifier = nn.Sequential(OrderedDict([('0', nn.Linear(64, EMBEDDING_SIZE, bias = False)), ('1', nn.Softmax())]))
        self.loss_function = nn.CrossEntropyLoss(weight=class_weights, reduce='mean')

    def forward(self, x):
        x = x.reshape(x.shape[0], 2, x.shape[1]//2)
        input_ids, attention_mask = x[:,0,:], x[:,1,:]
        output = self.features(input_ids, attention_mask=attention_mask)
        output = self.embeddings(output.pooler_output)
        output = self.classifier(output)
          
        return output

    def predict(self, x):
        x = self.forward(x)
        x = torch.argmax(x, dim=1)

        return x

class Architecture(torch.nn.Module):
    def __init__(self, model_select = "both", EMBEDDING_SIZE = 5, pre_train = True, class_weights = None):
        super().__init__()
        self.EMBEDDING_SIZE = EMBEDDING_SIZE
        
        # Audio
        audio_model = Audio_model(pre_train = True).features
        for param in audio_model[:11].parameters():
              param.requires_grad = False
        for param in audio_model[11:].parameters():
              param.requires_grad = True
        self.audio_model = audio_model

        # Text
        text_model = Text_model().features
        for param in text_model.parameters():
              param.requires_grad = True
        self.text_model = text_model

        # Embeddings
        self.embedding_audio = nn.Sequential(OrderedDict([('0', nn.Linear(32*NUM_MELS, 512)), ('1', nn.ReLU()), ('2', nn.Linear(512, 512)), ('3', nn.ReLU())]))
        self.embedding_text = nn.Sequential(OrderedDict([('0', nn.Linear(self.text_model.config.hidden_size, 512)), ('1', nn.ReLU()), ('2', nn.Linear(512, 512)), ('3', nn.ReLU())]))
        self.embedding_both = nn.Sequential(OrderedDict([('0', nn.Linear(32*NUM_MELS+self.text_model.config.hidden_size, 512)), ('1', nn.ReLU()), ('2', nn.Linear(512, 512)), ('3', nn.ReLU())]))

        # Classifier
        self.classifier = nn.Sequential(OrderedDict([('0', nn.Linear(512, EMBEDDING_SIZE, bias = False)), ('1', nn.Softmax())]))

        # Others parameters
        self.loss_function = nn.CrossEntropyLoss(weight=class_weights, reduce='mean')
        self.num_mels = NUM_MELS
        self.num_frames = NUM_FRAMES
        self.model_select = model_select

    def forward(self, x):
        # Audio
        x_audio = x[:, 0, :(self.num_frames*self.num_mels)]
        x_audio = x_audio.reshape(x.shape[0], 1, self.num_frames, self.num_mels)
        x_audio = self.audio_model(x_audio)
        x_audio = x_audio.mean(axis=-2).flatten(-2,-1)
        
        # Text
        x_text = x[:, 0, self.num_frames*self.num_mels:].to(torch.int)
        x_text = x_text.reshape(x_text.shape[0], 1, 2, x_text.shape[1]//2)
        input_ids, attention_mask = x_text[:,0,0,:], x_text[:,0,1,:]
        x_text = self.text_model(input_ids, attention_mask=attention_mask)
        
        if self.model_select=="audio":
          x = self.embedding_audio(x_audio)
          x = self.classifier(x)
        elif self.model_select=="text":
          x = self.embedding_text(x_text.pooler_output)
          x = self.classifier(x)
        else:
          x = torch.cat((x_audio, x_text.pooler_output), dim=1)
          x = self.embedding_both(x)
          x = self.classifier(x)

        return x


    def predict(self, x):
        x = self.forward(x)
        x = torch.argmax(x, dim=1)

        return x
    

class Process_independent_input(torch.nn.Module):
  def __init__(self):
    super().__init__()
    

  def get_pre_processed_input(self, text_data, audio_data, audio_sr):
      audio_class = Preprocess_audio(audio_data)
      text_class = Preprocess_text(text_data)
      data = []
      for i in range(text_data.shape[0]):
        audio = audio_class.wave_to_examples(audio_data[i], audio_sr, True, True).flatten()
        inputs_ids, attention_mask = text_class.__getitem__(i)
        data.append(torch.cat((audio, inputs_ids, attention_mask)))
      concatenation = torch.row_stack(data).reshape(text_data.shape[0], 1, -1).to(device)

      return concatenation 

  def make_audio_chunks(self, wav_data, sr):
    size = wav_data.shape[0]
    # Convert to mono.
    if len(wav_data.shape) > 1:
      wav_data = np.mean(wav_data, axis=1)
    if size>=(sr*EXAMPLE_SIZE):
      data = []
      for i in range(size//sr):
        new_data = wav_data[i*sr*EXAMPLE_SIZE:(i+1)*sr*EXAMPLE_SIZE]
        if new_data.shape[0]>sr:
          data.append(new_data)
      data[-1] = np.concatenate((data[-1], np.zeros(sr*EXAMPLE_SIZE - data[-1].shape[0])))
      data = np.row_stack(data)
    else:      
      data = np.concatenate((wav_data, np.zeros(sr*EXAMPLE_SIZE-size))).reshape(1, -1)
    return data
  
  def Save_List_Of_Audios_Chunks_In_DirOfSave_And_Return_PathList(self, data, dir_of_save, sr):
    counter = 0
    for audio in data:
      sf.write(dir_of_save + 'audio' + str(counter) + '.wav', audio, sr)
      counter += 1
    audio_paths = []
    for file in os.listdir(dir_of_save):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_of_save, file)):
            path_to_file = dir_of_save + file
        audio_paths.append(path_to_file)
    return audio_paths   

  def predict_emotions(self,audio_path, audio_data, audio_sr, dir_of_save):
      EMOTIONS = {emo: i for i, emo in enumerate(["Neutral", "Frustrated", "Angry", "Sad", "Excited"])}
      inverse_dict_emotions = {v: k for k, v in EMOTIONS.items()}
      model = torch.load('./best_model.zip').to(device).eval()

      original_Text = modelSpeechRecognition.transcribe([audio_path])[0]["transcription"]
      audio_chunks = self.make_audio_chunks(audio_data, audio_sr)
      Path_List = self.Save_List_Of_Audios_Chunks_In_DirOfSave_And_Return_PathList(audio_chunks,dir_of_save,audio_sr)
      text_chunks = []
      transcription = modelSpeechRecognition.transcribe(Path_List)
      counter = 0
      for path in Path_List:
        text_chunks.append(transcription[counter]["transcription"])
        print(transcription[counter]["transcription"])
        counter += 1

      text_chunks = np.array((text_chunks))
      input = self.get_pre_processed_input(text_chunks, audio_chunks, audio_sr).to(device)
      output = model(input).detach().cpu().numpy()

      predicts = {inverse_dict_emotions[i]:str(output[:,i].mean()) for i in range(len(EMOTIONS.items()))}   

      return predicts,original_Text

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the request
    file = request.files.get('file')

    if not file:
      return jsonify({"error": "No file provided"})
    
    # Save the file to disk
    file_path = './data/audio.wav'
    file.save(file_path)

    #audio_path = "projectst7-backend/audioNotWorking.wav"
    #audio_path = "projectst7-backend/audioWorking.wav"

    dir_of_save = "./data/split_audio/"
    audio, sr = librosa.load(file_path, sr = 16000)  

    try:
      result = Process_independent_input().predict_emotions(file_path,audio, sr, dir_of_save)
      response = {"prediction": result}
    except Exception as err:
      response = {"error": f"Error while predicting emotions: {err}"}

    # Clean files
    if os.path.isfile(file_path):
      os.remove(file_path)
    
    folder_path = "./data/split_audio/"
    file_list = os.listdir(folder_path)

    # Iterate through the list and remove each file
    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    response = jsonify(response)
    return response

@app.route('/', methods=['GET'])
def health():
    # Return a simple health check message
    response = jsonify({'message': 'Server is up and running'})
    return response

if __name__ == '__main__':
    # Create the "data" folder if it doesn't exist
    app.run()

if not os.path.exists("./data"):
      os.makedirs("./data")

# Create the "split_audio" subfolder if it doesn't exist
if not os.path.exists("./data/split_audio"):
    os.makedirs("./data/split_audio")