/**
 * Inference API Wrapper
 * Handles all communication with the backend server
 */

const API_BASE_URL = `http://${window.location.hostname}:8000/api`;
const WS_BASE_URL = `ws://${window.location.hostname}:8000/api`;

class InferenceAPI {
    constructor() {
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }

    /**
     * Get model information
     */
    async getModelInfo() {
        try {
            const response = await fetch(`${API_BASE_URL}/model-info`);
            if (!response.ok) throw new Error('Failed to get model info');
            return await response.json();
        } catch (error) {
            console.error('Error getting model info:', error);
            throw error;
        }
    }

    /**
     * Run inference on a single image (base64)
     */
    async inferImage(imageBase64, options = {}) {
        try {
            const response = await fetch(`${API_BASE_URL}/infer`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image_base64: imageBase64,
                    return_overlay: options.returnOverlay !== false,
                    return_comparison: options.returnComparison || false
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Inference failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Error during inference:', error);
            throw error;
        }
    }

    /**
     * Run inference on an uploaded image file
     */
    async inferFile(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_BASE_URL}/infer-file`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'File inference failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Error during file inference:', error);
            throw error;
        }
    }

    /**
     * Connect to WebSocket for streaming inference
     */
    connectWebSocket(onMessage, onError, onClose) {
        return new Promise((resolve, reject) => {
            try {
                this.websocket = new WebSocket(`${WS_BASE_URL}/ws/infer`);

                this.websocket.onopen = () => {
                    console.log('WebSocket connected');
                    this.reconnectAttempts = 0;
                    resolve(this.websocket);
                };

                this.websocket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (onMessage) onMessage(data);
                    } catch (error) {
                        console.error('Error parsing WebSocket message:', error);
                    }
                };

                this.websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    if (onError) onError(error);
                };

                this.websocket.onclose = () => {
                    console.log('WebSocket disconnected');
                    if (onClose) onClose();
                    this.attemptReconnect(onMessage, onError, onClose);
                };
            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Attempt to reconnect to WebSocket
     */
    attemptReconnect(onMessage, onError, onClose) {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            
            setTimeout(() => {
                this.connectWebSocket(onMessage, onError, onClose).catch(console.error);
            }, this.reconnectDelay * this.reconnectAttempts);
        }
    }

    /**
     * Send frame through WebSocket for inference
     */
    sendFrameForInference(imageBase64) {
        if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected');
            return false;
        }

        try {
            this.websocket.send(JSON.stringify({
                type: 'frame',
                image_base64: imageBase64
            }));
            return true;
        } catch (error) {
            console.error('Error sending frame:', error);
            return false;
        }
    }

    /**
     * Request model info through WebSocket
     */
    requestModelInfo() {
        if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected');
            return false;
        }

        try {
            this.websocket.send(JSON.stringify({
                type: 'info'
            }));
            return true;
        } catch (error) {
            console.error('Error requesting model info:', error);
            return false;
        }
    }

    /**
     * Stop WebSocket connection
     */
    closeWebSocket() {
        if (this.websocket) {
            this.websocket.send(JSON.stringify({ type: 'stop' }));
            this.websocket.close();
            this.websocket = null;
        }
    }

    /**
     * Check if WebSocket is connected
     */
    isConnected() {
        return this.websocket && this.websocket.readyState === WebSocket.OPEN;
    }
}

// Export singleton instance
const inferenceAPI = new InferenceAPI();
