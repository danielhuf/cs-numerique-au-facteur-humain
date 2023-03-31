import { useState, React } from "react";
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';
import { Radar } from 'react-chartjs-2';
import { BiLoaderAlt } from 'react-icons/bi';

import ang from '../img/ang.png';
import exc from '../img/exc.png';
import fru from '../img/fru.png';
import neu from '../img/neu.png';
import sad from '../img/sad.png';
import getPrediction  from "../utils/getPrediction";

// import { MicRecorder } from 'mic-recorder';
import { getWaveBlob } from "webm-to-wav-converter";

import ReactMicRecord from 'react-mic-record';
import { BsRecordCircle } from 'react-icons/bs';
import { BsStopCircle } from 'react-icons/bs';
import { FiUpload } from 'react-icons/fi';
import { FiTrash2 } from 'react-icons/fi';

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

const images = {
  'Neutral': neu,
  'Excited': exc,
  'Sad': sad,
  'Angry': ang,
  'Frustrated': fru
}

export default function Home() {
  const [showAudio, setShowAudio] = useState(false);
  const [audioUrl, setAudioUrl] = useState('');
  const [audioFile, setAudioFile] = useState(null);

  const [prediction, setPrediction] = useState({});
  const [audioText, setAudioText] = useState('');
  const [loading, setLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [chartData, setChartData] = useState([]);

  const [error, setError] = useState("");

  
  const createAudioFile = async (blobData) => {
    const waveBlob = await getWaveBlob(blobData.blob, { sampleRate: 44100 });
    const file = new File([waveBlob], 'audio.wav', {
      type: 'audio/wav',
      lastModified: Date.now()
    });
    handleUpdates(file);
    console.log("created audio file");
  };

  const handleUpdates = (file) => {
    setAudioUrl(URL.createObjectURL(file));
    setAudioFile(file);
    setShowAudio(true);
  };

  const getPredictionData = async () => {
    setLoading(true);
    try {
      const data = await getPrediction(audioFile);
      updatePrediction(data[0]);
      setAudioText(data[1]);
    } catch (error) {
      setError("An error occured while processing your audio. Please check if your audio is longer than 3 seconds and try again.");
    }
    setLoading(false);
  };

  const updatePrediction = (data) => {
    Object.keys(data).forEach(key => data[key] = parseFloat(data[key]));
    const max = Math.max(...Object.values(data));
    const emotion = Object.keys(data).find(key => data[key] === max);
    setPrediction(emotion);
    setChartData([
      parseFloat(data.Neutral) * 100,
      parseFloat(data.Frustrated) * 100,
      parseFloat(data.Angry) * 100,
      parseFloat(data.Sad) * 100,
      parseFloat(data.Excited) * 100
    ]);
  };

  const resetData = () => {
    setAudioUrl('');
    setAudioFile(null);
    setShowAudio(false);
    setPrediction({});
    setAudioText('');
    setChartData([]);
    setError("");
  };

  const data = {
    labels: ['Neutral', 'Frustrated', 'Angry', 'Sad', 'Excited'],
    datasets: [
      {
        label: 'Percentage of emotion',
        data: chartData,
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 1,
      },
    ],
  };

  const options = {
    scale: {
      ticks: {
        beginAtZero: true,
        max: 100,
        min: 0,
        stepSize: 20
      }
    }
  };

  return (
    <div className="flex flex-col items-center justify-center bg-gradient-to-b from-indigo-400 via-purple-400 to-pink-400">
      <div className="card flex flex-col bg-slate-50 drop-shadow-lg items-center p-20 gap-10 mt-10 mb-10">
        <h1 className="text-3xl text-black font-bold text-gray-700">Audio & Text Emotion Detector</h1>
        <div className="collapse collapse-arrow outline rounded-xl w-full text-slate-700">
          <input type="checkbox" className="peer" /> 
          <div className="collapse-title font-bold">
            How to Use 
          </div>
          <div className="collapse-content"> 
          <div>
            <p>1. Click on the red button to start recording, or use the upload button to upload a .wav file</p>
            <p>2. Click on the red button again to stop recording</p>
            <p>3. Click on the Get Prediction button to get the prediction*</p>
            <p>4. Click on the trash button to delete the audio file</p>
          </div>
            <p className="mt-5">*The prediction will take a while!</p>
          </div>
        </div>
          <ReactMicRecord
            key={isRecording}
            record={isRecording}         // defaults -> false.  Set to true to begin recording
            visualSetting="sinewave"
            className="sound-wave border-2 rounded-xl w-full border-black"
            strokeColor="#8D8AF8"
            backgroundColor="#0E1117"
            audioBitsPerSecond={705600} // defaults -> 128000
            onStop={createAudioFile}        // callback to execute when audio stops recording
          />
        <div className="flex flex-row bg-white gap-10 border-2 p-4 border-black rounded-full">
            <button className="btn btn-circle bg-primary">
              <FiTrash2 className="text-2xl text-white transition" onClick={() => resetData()} />
            </button>

          { !isRecording ? (
            <button onClick={() => setIsRecording(true)}>
              <BsRecordCircle className="text-5xl text-red-500 transition hover:text-red-700" />
            </button>
          ) : (
            <button onClick={() => setIsRecording(false)}>
              <BsStopCircle className="text-5xl text-red-500 transition hover:text-red-700" />
            </button>
          )}
          <div className="flex flex-row gap-5 items-center">
            <label htmlFor="file-input" className="btn btn-primary btn-circle">
              <FiUpload className="text-xl text-white stroke-white" htmlFor="file-input"/>            
            </label>
            <input type="file" id="file-input" accept="audio/wav" className="hidden" onChange={(e) => handleUpdates(e.target.files[0])} />
          </div>
        </div>
        {showAudio && (
          <div className="flex flex-col gap-5 w-full items-center">
            <audio controls src={audioUrl} type='audio/wav'>
              <track kind="captions" />
            </audio>
            <div className="flex flex-row gap-5 items-center">
              <a href={audioUrl} download="audio.wav">
                <button className="btn btn-primary">
                  Download Audio
                </button>
              </a>
              <button className="btn btn-primary" onClick={getPredictionData}>
                <div className="flex flex-row gap-2 items-center justify-center">
                  { loading ? <BiLoaderAlt className="animate-spin" /> : null }
                  Get Prediction
                </div>
              </button>
            </div>
          </div>
        )}

        {error && (
          <div className="text-red-500 font-bold">
            {error}
          </div>
        )}
        <div className="text-slate-700 font-bold">
          Prediction Result:
        </div>
        <div className="text-slate-700"> 
          <div className="flex flex-row gap-1">
            <p>You just said:</p>
            <p className="text-red-500 font-bold">{audioText}</p>
          </div>
          <div className="flex flex-row gap-1 w-12 h-12">
            <p>Emotion:</p>
            <img src={images[prediction]} alt={prediction}/>
          </div>  
          <Radar data={data} options={options}/>
        </div>
      </div>
    </div>
  );
}
