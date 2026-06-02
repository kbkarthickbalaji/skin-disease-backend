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

let localStream = null;
let capturedBlob = null;

// Camera trigger buttons
const cameraBtn = document.getElementById('cameraBtn');
const cameraContainer = document.getElementById('cameraContainer');
const video = document.getElementById('video');
const captureBtn = document.getElementById('captureBtn');
const closeCameraBtn = document.getElementById('closeCameraBtn');

if (cameraBtn) {
    cameraBtn.onclick = function() {
        // Hide previous previews
        const previewContainer = document.getElementById('selectedPreviewContainer');
        if (previewContainer) previewContainer.style.display = 'none';
        capturedBlob = null;
        
        // Reset camera inputs
        const cameraInput = document.getElementById('cameraInput');
        if (cameraInput) cameraInput.value = "";

        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
                .then(stream => {
                    localStream = stream;
                    if (video) video.srcObject = stream;
                    if (cameraContainer) cameraContainer.style.display = 'flex';
                    cameraBtn.style.display = 'none';
                })
                .catch(err => {
                    console.error("Camera access failed:", err);
                    alert("Could not access camera. Please check permissions.");
                });
        } else {
            alert("WebRTC Live camera is not supported in this browser/context. Please use the 'Take Photo' button instead.");
        }
    };
}

if (closeCameraBtn) {
    closeCameraBtn.onclick = function() {
        stopCamera();
    };
}

function stopCamera() {
    if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
        localStream = null;
    }
    if (cameraContainer) cameraContainer.style.display = 'none';
    if (cameraBtn) cameraBtn.style.display = 'inline-block';
}

if (captureBtn) {
    captureBtn.onclick = function() {
        if (!localStream || !video) return;

        // Create virtual canvas to capture the frame
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 320;
        canvas.height = video.videoHeight || 240;
        const ctx = canvas.getContext('2d');
        
        // Draw the current video frame on canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert to blob and show preview
        canvas.toBlob(blob => {
            capturedBlob = blob;
            
            // Show in preview container
            const previewContainer = document.getElementById('selectedPreviewContainer');
            const previewImg = document.getElementById('selectedPreviewImg');
            if (previewImg) previewImg.src = URL.createObjectURL(blob);
            if (previewContainer) previewContainer.style.display = 'block';
            
            // Stop the camera stream now
            stopCamera();
        }, 'image/png');
    };
}

// This listens for the 'Start Analysis' button click
const uploadBtn = document.getElementById('uploadBtn');
if (uploadBtn) {
    uploadBtn.onclick = function() {
        let fileInput = document.getElementById('imageInput');
        let cameraInput = document.getElementById('cameraInput');
        
        let hasFile = (fileInput && fileInput.files.length > 0) || 
                      (cameraInput && cameraInput.files.length > 0) || 
                      capturedBlob;

        // Validation: Check if a file is selected OR if we have a captured camera image
        if (!hasFile) {
            alert("Please choose a skin lesion image first, or click a sample below!");
            return;
        }

        // Prepare the data to send to Flask
        let formData = new FormData();
        if (capturedBlob) {
            formData.append('file', capturedBlob, 'captured_lesion.png');
        } else if (cameraInput && cameraInput.files.length > 0) {
            formData.append('file', cameraInput.files[0]);
        } else if (fileInput && fileInput.files.length > 0) {
            formData.append('file', fileInput.files[0]);
        }

        // Show a "Processing" message
        uploadBtn.innerText = "Analyzing...";
        uploadBtn.disabled = true;

        runAnalysis(formData);
    };
}

// Handles loading one of the clinic sample images automatically
function loadSampleImage(url, filename) {
    const uploadBtn = document.getElementById('uploadBtn');
    if (uploadBtn) {
        uploadBtn.innerText = "Loading Sample...";
        uploadBtn.disabled = true;
    }

    // Show preview in upload section
    const previewContainer = document.getElementById('selectedPreviewContainer');
    const previewImg = document.getElementById('selectedPreviewImg');
    if (previewImg) previewImg.src = url;
    if (previewContainer) previewContainer.style.display = 'block';

    // Fetch local sample file to send to API
    fetch(url)
        .then(res => res.blob())
        .then(blob => {
            const file = new File([blob], filename, { type: 'image/png' });
            let formData = new FormData();
            formData.append('file', file);
            
            // Clear regular inputs & camera stream
            const fileInput = document.getElementById('imageInput');
            if (fileInput) fileInput.value = "";
            const cameraInput = document.getElementById('cameraInput');
            if (cameraInput) cameraInput.value = "";
            capturedBlob = null;
            
            if (uploadBtn) {
                uploadBtn.innerText = "Analyzing Sample...";
            }
            runAnalysis(formData);
        })
        .catch(err => {
            console.error("Failed to load sample image: ", err);
            alert("An error occurred loading the clinical sample.");
            resetButtonState();
        });
}

// Handles immediate preview when you select a local file
const imageInput = document.getElementById('imageInput');
if (imageInput) {
    imageInput.addEventListener('change', function(e) {
        if (e.target.files[0]) {
            capturedBlob = null; // Clear camera capture if they choose a new file
            // Clear camera input
            const cameraInput = document.getElementById('cameraInput');
            if (cameraInput) cameraInput.value = "";
            
            const reader = new FileReader();
            reader.onload = function() {
                const previewContainer = document.getElementById('selectedPreviewContainer');
                const previewImg = document.getElementById('selectedPreviewImg');
                if (previewImg) previewImg.src = reader.result;
                if (previewContainer) previewContainer.style.display = 'block';
            };
            reader.readAsDataURL(e.target.files[0]);
        }
    });
}

// Handles immediate preview for native mobile camera input
const cameraInput = document.getElementById('cameraInput');
if (cameraInput) {
    cameraInput.addEventListener('change', function(e) {
        if (e.target.files[0]) {
            capturedBlob = null; // Clear WebRTC camera capture
            // Clear regular file input
            const fileInput = document.getElementById('imageInput');
            if (fileInput) fileInput.value = "";

            const reader = new FileReader();
            reader.onload = function() {
                const previewContainer = document.getElementById('selectedPreviewContainer');
                const previewImg = document.getElementById('selectedPreviewImg');
                if (previewImg) previewImg.src = reader.result;
                if (previewContainer) previewContainer.style.display = 'block';
            };
            reader.readAsDataURL(e.target.files[0]);
        }
    });
}