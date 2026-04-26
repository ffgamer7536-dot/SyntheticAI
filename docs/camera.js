/**
 * Webcam Capture Module
 * Handles camera stream acquisition and frame capture
 */

class CameraCapture {
    constructor(videoElement) {
        this.videoElement = videoElement;
        this.stream = null;
        this.isRunning = false;
        this.frameListener = null;
    }

    /**
     * Request camera permissions and start stream
     */
    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment'
                },
                audio: false
            });

            this.videoElement.srcObject = this.stream;
            this.isRunning = true;

            return new Promise((resolve) => {
                this.videoElement.onloadedmetadata = () => {
                    this.videoElement.play();
                    resolve();
                };
            });
        } catch (error) {
            console.error('Error accessing camera:', error);
            throw new Error(error.message || 'Camera access denied');
        }
    }

    /**
     * Stop camera stream
     */
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        this.isRunning = false;
        this.frameListener = null;
    }

    /**
     * Capture current frame as canvas
     */
    captureFrame() {
        if (!this.isRunning) {
            throw new Error('Camera not running');
        }

        const canvas = document.createElement('canvas');
        canvas.width = this.videoElement.videoWidth;
        canvas.height = this.videoElement.videoHeight;

        const ctx = canvas.getContext('2d');
        ctx.drawImage(this.videoElement, 0, 0);

        return canvas;
    }

    /**
     * Capture frame as base64 string
     */
    captureFrameBase64(format = 'jpeg', quality = 0.8) {
        const canvas = this.captureFrame();
        return canvas.toDataURL(`image/${format}`, quality).split(',')[1];
    }

    /**
     * Start continuous frame capture at specified interval
     */
    startFrameCapture(callback, intervalMs = 100) {
        if (!this.isRunning) {
            throw new Error('Camera not running');
        }

        this.frameListener = setInterval(() => {
            try {
                const frameBase64 = this.captureFrameBase64('jpeg', 0.85);
                callback(frameBase64);
            } catch (error) {
                console.error('Error capturing frame:', error);
            }
        }, intervalMs);
    }

    /**
     * Stop continuous frame capture
     */
    stopFrameCapture() {
        if (this.frameListener) {
            clearInterval(this.frameListener);
            this.frameListener = null;
        }
    }

    /**
     * Get current video dimensions
     */
    getDimensions() {
        if (!this.isRunning) {
            return null;
        }
        return {
            width: this.videoElement.videoWidth,
            height: this.videoElement.videoHeight
        };
    }

    /**
     * Draw canvas overlay on video
     */
    drawOverlay(canvas, targetCanvas) {
        if (!this.isRunning) {
            throw new Error('Camera not running');
        }

        targetCanvas.width = this.videoElement.videoWidth;
        targetCanvas.height = this.videoElement.videoHeight;

        const ctx = targetCanvas.getContext('2d');
        ctx.drawImage(this.videoElement, 0, 0);
        ctx.drawImage(canvas, 0, 0);
    }

    /**
     * Draw blended segmentation overlay
     */
    drawBlendedOverlay(segmentationCanvas, targetCanvas, alpha = 0.6) {
        if (!this.isRunning) {
            throw new Error('Camera not running');
        }

        targetCanvas.width = this.videoElement.videoWidth;
        targetCanvas.height = this.videoElement.videoHeight;

        const ctx = targetCanvas.getContext('2d');

        // Draw video frame
        ctx.drawImage(this.videoElement, 0, 0);

        // Draw segmentation with alpha blending
        ctx.globalAlpha = alpha;
        ctx.drawImage(segmentationCanvas, 0, 0);
        ctx.globalAlpha = 1.0;
    }

    /**
     * Get supported video formats and constraints
     */
    static async getSupportedConstraints() {
        const constraints = {
            video: {
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            }
        };

        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            
            return {
                supported: videoDevices.length > 0,
                devices: videoDevices,
                constraints: constraints
            };
        } catch (error) {
            console.error('Error getting device info:', error);
            return {
                supported: false,
                devices: [],
                constraints: constraints
            };
        }
    }
}

// Helper function to initialize camera
async function initializeCamera(videoElement) {
    const camera = new CameraCapture(videoElement);
    await camera.start();
    return camera;
}
