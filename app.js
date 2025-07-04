let model;
let audioContext;
let micStream;
let meydaAnalyzer;
let isListening = false;
const labels = ["kick", "snare", "hat"];

async function loadModel() {
  if (!model) {
    model = await tf.loadGraphModel("model/model.json");
    console.log("Model loaded");
  }
}

async function toggleListening() {
  if (isListening) {
    stopListening();
    document.getElementById("listenBtn").innerText = "Start Listening";
    document.getElementById("status").innerText = "Stopped";
  } else {
    await startListening();
    document.getElementById("listenBtn").innerText = "Stop Listening";
    document.getElementById("status").innerText = "Listening...";
  }
  isListening = !isListening;
}

async function startListening() {
  await loadModel();

  audioContext = new AudioContext();
  micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const source = audioContext.createMediaStreamSource(micStream);

  meydaAnalyzer = Meyda.createMeydaAnalyzer({
    audioContext,
    source,
    bufferSize: 512,
    featureExtractors: ['mfcc'],
    callback: async (features) => {
      const mfcc = features.mfcc;
      const input = tf.tensor(mfcc).reshape([1, mfcc.length]);
      const prediction = model.predict(input);
      const result = await prediction.argMax(1).data();
      const label = labels[result[0]];
      playSound(label);
      console.log("Heard:", label);
    }
  });

  meydaAnalyzer.start();
}

function stopListening() {
  if (meydaAnalyzer) meydaAnalyzer.stop();
  if (micStream) {
    const tracks = micStream.getTracks();
    tracks.forEach(track => track.stop());
  }
  if (audioContext && audioContext.state !== "closed") {
    audioContext.close();
  }
}

function playSound(label) {
  const audio = document.getElementById(label);
  if (audio) {
    audio.currentTime = 0;
    audio.play();
  }
}
