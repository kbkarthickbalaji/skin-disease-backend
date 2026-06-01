let chartInstance = null;

// Reusable analysis function
function runAnalysis(formData) {
    // Send the image to the /predict route in app.py
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
            resetButtonState();
            return;
        }

        // Unhide the result container
        document.getElementById('result-container').style.display = 'block';
        
        // Fill in the Text Data
        document.getElementById('previewImg').src = data.image_url;
        document.getElementById('resClass').innerText = data.class;
        document.getElementById('resConf').innerText = data.confidence;
        document.getElementById('resRemedy').innerText = data.remedy;
        document.getElementById('resDoctor').innerText = data.doctor;

        // Create/Update the Probability Chart
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

        resetButtonState();
        
        // Scroll down to the results automatically
        document.getElementById('result-container').scrollIntoView({ behavior: 'smooth' });
    })
    .catch(error => {
        console.error('Error:', error);
        alert("An error occurred during analysis.");
        resetButtonState();
    });
}

function resetButtonState() {
    document.getElementById('uploadBtn').innerText = "Start Analysis";
    document.getElementById('uploadBtn').disabled = false;
}

// This listens for the 'Start Analysis' button click
document.getElementById('uploadBtn').onclick = function() {
    let fileInput = document.getElementById('imageInput');
    
    // Validation: Check if a file is selected
    if (fileInput.files.length === 0) {
        alert("Please choose a skin lesion image first, or click a sample below!");
        return;
    }

    // Prepare the data to send to Flask
    let formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Show a "Processing" message
    document.getElementById('uploadBtn').innerText = "Analyzing...";
    document.getElementById('uploadBtn').disabled = true;

    runAnalysis(formData);
};

// Handles loading one of the clinic sample images automatically
function loadSampleImage(url, filename) {
    document.getElementById('uploadBtn').innerText = "Loading Sample...";
    document.getElementById('uploadBtn').disabled = true;

    // Show preview in upload section
    const previewContainer = document.getElementById('selectedPreviewContainer');
    const previewImg = document.getElementById('selectedPreviewImg');
    previewImg.src = url;
    previewContainer.style.display = 'block';

    // Fetch local sample file to send to API
    fetch(url)
        .then(res => res.blob())
        .then(blob => {
            const file = new File([blob], filename, { type: 'image/png' });
            let formData = new FormData();
            formData.append('file', file);
            
            // Clear regular input
            document.getElementById('imageInput').value = "";
            
            document.getElementById('uploadBtn').innerText = "Analyzing Sample...";
            runAnalysis(formData);
        })
        .catch(err => {
            console.error("Failed to load sample image: ", err);
            alert("An error occurred loading the clinical sample.");
            resetButtonState();
        });
}

// This part handles the immediate image preview when you "Choose File"
document.getElementById('imageInput').addEventListener('change', function(e) {
    if (e.target.files[0]) {
        const reader = new FileReader();
        reader.onload = function() {
            const previewContainer = document.getElementById('selectedPreviewContainer');
            const previewImg = document.getElementById('selectedPreviewImg');
            previewImg.src = reader.result;
            previewContainer.style.display = 'block';
        };
        reader.readAsDataURL(e.target.files[0]);
    }
});