document.getElementById('predictForm').onsubmit = async function(e) {
  e.preventDefault();
  // Collect feature values from form
  const form = e.target;
  const features = [
    Number(form.amount.value),
    Number(form.time.value),
    Number(form.v1.value),
    Number(form.v2.value)
    // ...add all features in order
  ];
  const res = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({features})
  });
  const data = await res.json();
  document.getElementById('result').innerText = 'Prediction: ' + (data.prediction ? 'Default' : 'No Default');
};
