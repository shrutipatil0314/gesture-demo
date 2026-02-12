const video = document.getElementById('webcam');
const caption = document.getElementById('caption');

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  });

// Placeholder gesture recognition
// Replace with MediaPipe Hands JS or TensorFlow.js model
function mockGestureRecognition() {
  const gestures = ["Hello ✋", "Yes 👍", "No 👎", "Peace ✌️", "Okay 👌"];
  const randomGesture = gestures[Math.floor(Math.random() * gestures.length)];
  caption.innerText = randomGesture;
}

// Demo: update every 3 seconds
setInterval(mockGestureRecognition, 3000);