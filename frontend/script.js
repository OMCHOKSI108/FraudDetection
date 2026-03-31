
document.getElementById('predictForm').onsubmit = async function(e) {
    e.preventDefault();
    const form = e.target;
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');
    
    resultDiv.style.display = 'none';
    loadingDiv.style.display = 'block';
    
    const features = {};
    for (let i = 1; i <= 28; i++) {
        features[`V${i}`] = Number(form[`v${i}`]?.value || 0);
    }
    
    features['Amount'] = Number(form.amount.value);
    features['Time'] = Number(form.time.value);
    
    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({features})
        });
        const data = await res.json();
        
        loadingDiv.style.display = 'none';
        resultDiv.style.display = 'block';
        
        if (data.prediction === 1) {
            resultDiv.className = 'fraud';
            resultDiv.innerHTML = `<strong>FRAUD DETECTED!</strong><br>Confidence: ${(data.probability * 100).toFixed(2)}%`;
        } else {
            resultDiv.className = 'legit';
            resultDiv.innerHTML = `<strong>LEGITIMATE TRANSACTION</strong><br>Confidence: ${((1 - data.probability) * 100).toFixed(2)}%`;
        }
    } catch (error) {
        loadingDiv.style.display = 'none';
        resultDiv.style.display = 'block';
        resultDiv.className = 'error';
        resultDiv.innerHTML = `Error: ${error.message}`;
    }
};
