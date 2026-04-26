/**
 * Main Application Logic
 * Coordinates all UI interactions and API calls
 */

let currentCamera = null;
let isWebcamProcessing = false;
let lastFrameTime = 0;
let frameCount = 0;
let fps = 0;

// =============================================================================
// TAB SWITCHING
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    initializeColorLegend();
});

function initializeEventListeners() {
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            switchTab(e.target.dataset.tab);
        });
    });

    // Upload Image Tab
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const inferBtn = document.getElementById('inferBtn');

    uploadArea.addEventListener('click', () => imageInput.click());
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            imageInput.files = e.dataTransfer.files;
            inferBtn.disabled = false;
        }
    });

    imageInput.addEventListener('change', () => {
        if (imageInput.files.length > 0) {
            inferBtn.disabled = false;
        }
    });

    inferBtn.addEventListener('click', () => inferImage());

    // Upload Video Tab
    const uploadVideoArea = document.getElementById('uploadVideoArea');
    const videoInput = document.getElementById('videoInput');
    const processVideoBtn = document.getElementById('processVideoBtn');

    uploadVideoArea.addEventListener('click', () => videoInput.click());
    uploadVideoArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadVideoArea.classList.add('dragover');
    });
    uploadVideoArea.addEventListener('dragleave', () => uploadVideoArea.classList.remove('dragover'));
    uploadVideoArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadVideoArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            videoInput.files = e.dataTransfer.files;
            processVideoBtn.disabled = false;
        }
    });

    videoInput.addEventListener('change', () => {
        if (videoInput.files.length > 0) {
            processVideoBtn.disabled = false;
        }
    });

    processVideoBtn.addEventListener('click', () => processVideo());

    // Webcam Tab
    const startWebcamBtn = document.getElementById('startWebcamBtn');
    const stopWebcamBtn = document.getElementById('stopWebcamBtn');

    startWebcamBtn.addEventListener('click', () => startWebcam());
    stopWebcamBtn.addEventListener('click', () => stopWebcam());
}

function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }

    // Mark button as active
    event.target.classList.add('active');

    // Clean up if switching away from webcam
    if (tabName !== 'webcam' && currentCamera) {
        stopWebcam();
    }
}

function initializeColorLegend() {
    segmentationViewer.createLegend();
}

// =============================================================================
// IMAGE INFERENCE
// =============================================================================

async function inferImage() {
    const imageInput = document.getElementById('imageInput');
    if (!imageInput.files.length) return;

    const file = imageInput.files[0];
    const errorDiv = document.getElementById('imageError');
    const loadingDiv = document.getElementById('loadingSpinner');
    const resultsDiv = document.getElementById('imageResults');

    errorDiv.style.display = 'none';
    loadingDiv.style.display = 'flex';
    resultsDiv.style.display = 'none';

    try {
        // Read image file as base64
        const reader = new FileReader();
        const imageBase64 = await new Promise((resolve, reject) => {
            reader.onload = (e) => {
                const base64 = e.target.result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });

        // Infer
        const results = await inferenceAPI.inferImage(imageBase64);

        // Display results
        segmentationViewer.displayResults(
            document.getElementById('originalCanvas'),
            document.getElementById('predictionCanvas'),
            imageBase64,
            results.image_base64
        );

        statsPanel.displayResults(results);

        resultsDiv.style.display = 'block';
        loadingDiv.style.display = 'none';

    } catch (error) {
        errorDiv.textContent = `Error: ${error.message}`;
        errorDiv.style.display = 'block';
        loadingDiv.style.display = 'none';
        console.error('Inference error:', error);
    }
}

// =============================================================================
// VIDEO PROCESSING
// =============================================================================

async function processVideo() {
    const videoInput = document.getElementById('videoInput');
    if (!videoInput.files.length) return;

    const file = videoInput.files[0];
    const errorDiv = document.getElementById('videoError');
    const progressDiv = document.getElementById('videoProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');

    errorDiv.style.display = 'none';
    progressDiv.style.display = 'block';

    try {
        // Read video file
        const reader = new FileReader();
        const videoBlob = await new Promise((resolve, reject) => {
            reader.onload = (e) => resolve(new Blob([e.target.result]));
            reader.onerror = reject;
            reader.readAsArrayBuffer(file);
        });

        // For now, just show a message that video processing requires backend implementation
        progressDiv.style.display = 'none';
        errorDiv.textContent = 'Video processing coming soon. Backend implementation required.';
        errorDiv.style.display = 'block';

    } catch (error) {
        progressDiv.style.display = 'none';
        errorDiv.textContent = `Error: ${error.message}`;
        errorDiv.style.display = 'block';
        console.error('Video processing error:', error);
    }
}

// =============================================================================
// WEBCAM INFERENCE
// =============================================================================

async function startWebcam() {
    const webcamVideo = document.getElementById('webcamVideo');
    const startBtn = document.getElementById('startWebcamBtn');
    const stopBtn = document.getElementById('stopWebcamBtn');
    const errorDiv = document.getElementById('webcamError');
    const statsDiv = document.getElementById('webcamStats');

    errorDiv.style.display = 'none';

    try {
        // Initialize camera
        currentCamera = await initializeCamera(webcamVideo);
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statsDiv.style.display = 'grid';

        // Start streaming
        startWebSocketInference();

    } catch (error) {
        errorDiv.textContent = `Camera Error: ${error.message}`;
        errorDiv.style.display = 'block';
        currentCamera = null;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        console.error('Webcam start error:', error);
    }
}

async function startWebSocketInference() {
    const autoStream = document.getElementById('autoStream');
    const showOverlay = document.getElementById('showOverlay');

    if (!autoStream.checked) {
        console.log('Auto-stream disabled');
        return;
    }

    try {
        // Connect to WebSocket
        await inferenceAPI.connectWebSocket(
            (data) => handleWebSocketMessage(data, showOverlay),
            console.error,
            () => console.log('WebSocket closed')
        );

        // Start capturing frames
        const captureInterval = 100; // ~10 FPS
        currentCamera.startFrameCapture(async (frameBase64) => {
            inferenceAPI.sendFrameForInference(frameBase64);
        }, captureInterval);

    } catch (error) {
        console.error('WebSocket error:', error);
    }
}

function handleWebSocketMessage(data, showOverlay) {
    if (data.type === 'results') {
        const now = performance.now();
        const timeDelta = (now - lastFrameTime) / 1000;
        lastFrameTime = now;
        frameCount++;

        if (frameCount % 10 === 0) {
            fps = 10 / timeDelta;
        }

        // Update display
        const liveCanvas = document.getElementById('liveCanvas');
        const resultsDiv = document.getElementById('webcamResults');

        segmentationViewer.drawSegmentationFrame(
            liveCanvas,
            data.predictions_base64,
            null,
            false
        );

        if (data.coverage) {
            statsPanel.displayLiveCoverage(data.coverage);
        }

        if (data.inference_time_ms) {
            statsPanel.updateWebcamStats(data.inference_time_ms, fps, null);
        }

        resultsDiv.style.display = 'block';

    } else if (data.type === 'error') {
        console.error('Inference error:', data.message);
    }
}

function stopWebcam() {
    if (currentCamera) {
        currentCamera.stopFrameCapture();
        currentCamera.stop();
        currentCamera = null;
    }

    inferenceAPI.closeWebSocket();

    const startBtn = document.getElementById('startWebcamBtn');
    const stopBtn = document.getElementById('stopWebcamBtn');
    const resultsDiv = document.getElementById('webcamResults');

    startBtn.disabled = false;
    stopBtn.disabled = true;
    resultsDiv.style.display = 'none';
    frameCount = 0;
    fps = 0;
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function showError(message, containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.style.display = 'block';
        container.textContent = message;
    }
}

function showLoading(shown, containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.style.display = shown ? 'flex' : 'none';
    }
}
