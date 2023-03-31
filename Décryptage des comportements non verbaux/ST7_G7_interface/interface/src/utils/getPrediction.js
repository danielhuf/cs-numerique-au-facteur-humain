import { config } from '../Constants';

const getPrediction = async (audioFile) => {
  const formData = new FormData();
  formData.append('file', audioFile);

  const response = await fetch(`${config.url.API_URL}/predict`, {
    method: 'POST',
    body: formData,
  });
  const data = await response.json();
  console.log(data)

  if (response.status !== 200) {
    throw Error(data.message);
  } else if (!data.hasOwnProperty('prediction')) {
    throw Error(data.error);
  }
  return data.prediction;
}

export default getPrediction;