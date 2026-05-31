let chartInstance = null;

// This listens for the 'Start Analysis' button click
document.getElementById('uploadBtn').onclick = function() {
    let fileInput = document.getElementById('imageInput');
    
    // 1. Validation: Check if a file is selected
    if (fileInput.files.length === 0) {
        alert("Please choose a skin lesion image first!");
        return;
    }

    // 2. Prepare the data to send to Flask
    let formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // 3. Show a "Processing" message (Optional but professional)
    document.getElementById('uploadBtn').innerText = "Analyzing...";
    document.getElementById('uploadBtn').disabled = true;

    // 4. Send the image to the /predict route in app.py
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // 5. Unhide the result container
        document.getElementById('result-container').style.display = 'block';
        
        // 6. Fill in the Text Data
        document.getElementById('previewImg').src = data.image_url;
        document.getElementById('resClass').innerText = data.class;
        document.getElementById('resConf').innerText = data.confidence;
        document.getElementById('resRemedy').innerText = data.remedy;
        document.getElementById('resDoctor').innerText = data.doctor;

        // 7. Create/Update the Probability Chart
        const ctx = document.getElementById('predictionChart').getContext('2d');
        
        // If a chart already exists, destroy it before making a new one
        if (chartInstance) {
            chartInstance.destroy();
        }
        
        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.chart_labels || ['AK', 'BCC', 'DF', 'BKL', 'MEL', 'NV', 'VASC'],
                datasets: [{
                    label: 'Probability %',
                    data: data.chart_data.map(val => (val * 100).toFixed(2)),
                    backgroundColor: '#2d9d8f',
                    borderRadius: 5
                }]
            },
            options: {
                indexAxis: 'y', // Makes it a horizontal bar chart
                responsive: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { beginAtZero: true, max: 100 }
                }
            }
        });

        // Reset button state
        document.getElementById('uploadBtn').innerText = "Start Analysis";
        document.getElementById('uploadBtn').disabled = false;
        
        // Scroll down to the results automatically
        document.getElementById('result-container').scrollIntoView({ behavior: 'smooth' });
    })
    .catch(error => {
        console.error('Error:', error);
        alert("An error occurred during analysis.");
        document.getElementById('uploadBtn').innerText = "Start Analysis";
        document.getElementById('uploadBtn').disabled = false;
    });
};

// This part handles the immediate image preview when you "Choose File"
document.getElementById('imageInput').addEventListener('change', function(e) {
    if (e.target.files[0]) {
        const reader = new FileReader();
        reader.onload = function() {
            // If you have a preview element in your hero section, show it here
            console.log("Image selected and ready for analysis.");
        };
        reader.readAsDataURL(e.target.files[0]);
    }
});