import { createFFmpeg } from '@ffmpeg/ffmpeg/dist/ffmpeg.min.js';


async function convertWebmToWav(webmBlob) {
  const ffmpeg = createFFmpeg({
    corePath: 'https://unpkg.com/@ffmpeg/core@0.10.0/dist/ffmpeg-core.js',
      log: true,
  });
  await ffmpeg.load();

  const inputName = 'input.webm';
  const outputName = 'output.wav';

  ffmpeg.FS('writeFile', inputName, await fetch(webmBlob).then((res) => res.arrayBuffer()));

  await ffmpeg.run('-i', inputName, outputName);

  const outputData = ffmpeg.FS('readFile', outputName);
  const outputBlob = new Blob([outputData.buffer], { type: 'audio/wav' });

  return outputBlob;
}

export default convertWebmToWav;