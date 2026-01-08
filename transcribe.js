/**
 * Transcriptify - Client-side video transcription using Web Speech API
 * All processing happens in the browser - no server required
 */

class Transcriber {
    constructor(options = {}) {
        this.options = {
            language: options.language || 'en-US',
            continuous: options.continuous !== false,
            interimResults: options.interimResults !== false
        };
        
        this.recognition = null;
        this.isTranscribing = false;
        this.segments = [];
        this.currentText = '';
        this.listeners = {};
        this.audioContext = null;
        this.mediaElement = null;
    }

    /**
     * Check if Web Speech API is supported
     */
    isSupported() {
        return 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
    }

    /**
     * Add event listener
     */
    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
        return this;
    }

    /**
     * Remove event listener
     */
    off(event, callback) {
        if (this.listeners[event]) {
            this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
        }
        return this;
    }

    /**
     * Emit event
     */
    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => callback(data));
        }
    }

    /**
     * Transcribe a video/audio file
     */
    async transcribe(file, options = {}) {
        if (!this.isSupported()) {
            throw new Error('Web Speech API is not supported in this browser. Please use Chrome or Edge.');
        }

        const language = options.language || this.options.language;
        const onProgress = options.onProgress || (() => {});
        const onPartialResult = options.onPartialResult || (() => {});

        return new Promise((resolve, reject) => {
            // Reset state
            this.segments = [];
            this.currentText = '';
            this.isTranscribing = true;

            // Create media element
            const mediaUrl = URL.createObjectURL(file);
            this.mediaElement = document.createElement('video');
            this.mediaElement.src = mediaUrl;
            this.mediaElement.muted = true; // Mute to avoid echo

            // Wait for metadata
            this.mediaElement.onloadedmetadata = () => {
                const duration = this.mediaElement.duration;
                let startTime = 0;

                // Setup speech recognition
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                this.recognition = new SpeechRecognition();
                this.recognition.lang = language;
                this.recognition.continuous = true;
                this.recognition.interimResults = true;
                this.recognition.maxAlternatives = 1;

                let finalTranscript = '';
                let segmentStartTime = 0;

                this.recognition.onstart = () => {
                    this.emit('start');
                    segmentStartTime = this.mediaElement.currentTime;
                };

                this.recognition.onresult = (event) => {
                    let interimTranscript = '';

                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        const result = event.results[i];
                        const transcript = result[0].transcript;
                        
                        if (result.isFinal) {
                            finalTranscript += transcript + ' ';
                            
                            // Create segment
                            const segment = {
                                text: transcript.trim(),
                                startTime: segmentStartTime,
                                endTime: this.mediaElement.currentTime,
                                confidence: result[0].confidence || 0.9
                            };
                            
                            this.segments.push(segment);
                            segmentStartTime = this.mediaElement.currentTime;
                            
                            this.emit('result', { text: transcript, isFinal: true });
                        } else {
                            interimTranscript += transcript;
                        }
                    }

                    this.currentText = finalTranscript + interimTranscript;
                    onPartialResult(this.currentText);
                    this.emit('result', { text: this.currentText, isFinal: false });
                };

                this.recognition.onerror = (event) => {
                    // Ignore no-speech errors and aborted errors during normal operation
                    if (event.error === 'no-speech' || event.error === 'aborted') {
                        return;
                    }
                    
                    console.error('Speech recognition error:', event.error);
                    this.emit('error', { error: event.error });
                };

                this.recognition.onend = () => {
                    if (this.isTranscribing && this.mediaElement && !this.mediaElement.ended) {
                        // Restart recognition if still playing
                        try {
                            this.recognition.start();
                        } catch (e) {
                            // Recognition might already be starting
                        }
                    }
                };

                // Progress tracking
                this.mediaElement.ontimeupdate = () => {
                    const progress = Math.round((this.mediaElement.currentTime / duration) * 100);
                    onProgress(progress);
                    this.emit('progress', { progress });
                };

                // When video ends
                this.mediaElement.onended = () => {
                    this.isTranscribing = false;
                    
                    if (this.recognition) {
                        this.recognition.stop();
                    }

                    // Clean up
                    URL.revokeObjectURL(mediaUrl);

                    const result = {
                        text: finalTranscript.trim(),
                        segments: this.segments,
                        duration: duration,
                        language: language
                    };

                    this.emit('end', result);
                    resolve(result);
                };

                // Handle errors
                this.mediaElement.onerror = (e) => {
                    this.isTranscribing = false;
                    URL.revokeObjectURL(mediaUrl);
                    reject(new Error('Failed to load media file'));
                };

                // Start playback and recognition
                this.mediaElement.play().then(() => {
                    // Use audio capture for speech recognition
                    this.setupAudioCapture(this.mediaElement).then(() => {
                        try {
                            this.recognition.start();
                        } catch (e) {
                            console.error('Failed to start recognition:', e);
                        }
                    });
                }).catch(reject);
            };

            this.mediaElement.onerror = () => {
                reject(new Error('Failed to load media file'));
            };
        });
    }

    /**
     * Setup audio capture from media element
     */
    async setupAudioCapture(mediaElement) {
        try {
            // Request microphone permission - this is needed for speech recognition
            // The actual audio comes from the video, but we need permission
            await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (e) {
            console.warn('Microphone access denied. Speech recognition may not work properly.');
        }
    }

    /**
     * Cancel ongoing transcription
     */
    cancel() {
        this.isTranscribing = false;
        
        if (this.recognition) {
            this.recognition.stop();
            this.recognition = null;
        }
        
        if (this.mediaElement) {
            this.mediaElement.pause();
            this.mediaElement.src = '';
            this.mediaElement = null;
        }

        this.emit('cancel');
    }

    /**
     * Convert segments to SRT format
     */
    static toSRT(segments) {
        return segments.map((segment, index) => {
            const startTime = Transcriber.formatTimeSRT(segment.startTime);
            const endTime = Transcriber.formatTimeSRT(segment.endTime);
            return `${index + 1}\n${startTime} --> ${endTime}\n${segment.text}\n`;
        }).join('\n');
    }

    /**
     * Convert segments to WebVTT format
     */
    static toVTT(segments) {
        const lines = ['WEBVTT\n'];
        segments.forEach((segment, index) => {
            const startTime = Transcriber.formatTimeVTT(segment.startTime);
            const endTime = Transcriber.formatTimeVTT(segment.endTime);
            lines.push(`${index + 1}`);
            lines.push(`${startTime} --> ${endTime}`);
            lines.push(segment.text);
            lines.push('');
        });
        return lines.join('\n');
    }

    /**
     * Format time for SRT (HH:MM:SS,mmm)
     */
    static formatTimeSRT(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 1000);
        return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
    }

    /**
     * Format time for VTT (HH:MM:SS.mmm)
     */
    static formatTimeVTT(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 1000);
        return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
    }

    /**
     * Format time for display (MM:SS)
     */
    static formatTimeDisplay(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }
}

// ===== UI Code (only runs on main page) =====

document.addEventListener('DOMContentLoaded', () => {
    // Check if we're on the main page
    const dropzone = document.getElementById('dropzone');
    if (!dropzone) return;

    const fileInput = document.getElementById('fileInput');
    const processingSection = document.getElementById('processing-section');
    const resultSection = document.getElementById('result-section');
    const videoPlayer = document.getElementById('videoPlayer');
    const transcribeBtn = document.getElementById('transcribeBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const statusEl = document.getElementById('status');
    const progressContainer = document.getElementById('progress-container');
    const progressEl = document.getElementById('progress');
    const progressText = document.getElementById('progress-text');
    const transcriptEl = document.getElementById('transcript');
    const copyBtn = document.getElementById('copyBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const downloadSrtBtn = document.getElementById('downloadSrtBtn');

    const transcriber = new Transcriber();
    let currentFile = null;
    let transcriptionResult = null;

    // Check browser support
    if (!transcriber.isSupported()) {
        statusEl.textContent = '⚠️ Your browser does not support speech recognition. Please use Chrome or Edge.';
        statusEl.style.color = '#ff6b6b';
    }

    // Drag and drop handlers
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('video/')) {
            handleFile(files[0]);
        }
    });

    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        currentFile = file;
        
        // Show video preview
        const videoUrl = URL.createObjectURL(file);
        videoPlayer.src = videoUrl;
        
        // Show processing section
        processingSection.classList.remove('hidden');
        resultSection.classList.add('hidden');
        
        // Reset state
        statusEl.textContent = `Loaded: ${file.name}`;
        progressContainer.classList.add('hidden');
        progressEl.style.width = '0%';
        progressText.textContent = '0%';
        transcriptEl.textContent = '';
        
        transcribeBtn.disabled = false;
        cancelBtn.classList.add('hidden');
        
        // Scroll to processing section
        processingSection.scrollIntoView({ behavior: 'smooth' });
    }

    transcribeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        transcribeBtn.disabled = true;
        cancelBtn.classList.remove('hidden');
        progressContainer.classList.remove('hidden');
        resultSection.classList.remove('hidden');
        
        statusEl.textContent = 'Starting transcription... (microphone access required)';
        transcriptEl.innerHTML = '<span style="color: var(--text-muted);">Listening for speech...</span>';

        try {
            transcriptionResult = await transcriber.transcribe(currentFile, {
                onProgress: (progress) => {
                    progressEl.style.width = `${progress}%`;
                    progressText.textContent = `${progress}%`;
                },
                onPartialResult: (text) => {
                    if (text.trim()) {
                        transcriptEl.textContent = text;
                    }
                }
            });

            statusEl.textContent = 'Transcription complete!';
            cancelBtn.classList.add('hidden');
            
            // Display final result with timestamps
            displayTranscript(transcriptionResult);

        } catch (error) {
            console.error('Transcription error:', error);
            statusEl.textContent = `Error: ${error.message}`;
            statusEl.style.color = '#ff6b6b';
            transcribeBtn.disabled = false;
            cancelBtn.classList.add('hidden');
        }
    });

    cancelBtn.addEventListener('click', () => {
        transcriber.cancel();
        statusEl.textContent = 'Transcription cancelled';
        transcribeBtn.disabled = false;
        cancelBtn.classList.add('hidden');
        progressContainer.classList.add('hidden');
    });

    function displayTranscript(result) {
        if (!result.segments.length && !result.text) {
            transcriptEl.innerHTML = '<span style="color: var(--text-muted);">No speech detected in the video.</span>';
            return;
        }

        if (result.segments.length > 0) {
            transcriptEl.innerHTML = result.segments.map(segment => `
                <div class="segment">
                    <div class="timestamp">${Transcriber.formatTimeDisplay(segment.startTime)} - ${Transcriber.formatTimeDisplay(segment.endTime)}</div>
                    <div>${segment.text}</div>
                </div>
            `).join('');
        } else {
            transcriptEl.textContent = result.text;
        }
    }

    copyBtn.addEventListener('click', () => {
        const text = transcriptionResult ? transcriptionResult.text : transcriptEl.textContent;
        navigator.clipboard.writeText(text).then(() => {
            const originalHTML = copyBtn.innerHTML;
            copyBtn.innerHTML = '✓';
            setTimeout(() => {
                copyBtn.innerHTML = originalHTML;
            }, 2000);
        });
    });

    downloadBtn.addEventListener('click', () => {
        if (!transcriptionResult) return;
        
        const blob = new Blob([transcriptionResult.text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'transcript.txt';
        a.click();
        URL.revokeObjectURL(url);
    });

    downloadSrtBtn.addEventListener('click', () => {
        if (!transcriptionResult || !transcriptionResult.segments.length) return;
        
        const srt = Transcriber.toSRT(transcriptionResult.segments);
        const blob = new Blob([srt], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'transcript.srt';
        a.click();
        URL.revokeObjectURL(url);
    });
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Transcriber };
}
