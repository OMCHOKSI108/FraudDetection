document.getElementById('predictForm').onsubmit = async function(e) {
  e.preventDefault();
  // Collect feature values from form
  const form = e.target;
  const features = [
    Number(form.feature1.value),
    // ...add all features in order
  ];
  const res = await fetch('https://your-api-url.onrender.com/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({features})
  });
  const data = await res.json();
  document.getElementById('result').innerText = 'Prediction: ' + (data.prediction ? 'Default' : 'No Default');
};
