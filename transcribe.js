/**
 * Transcriptify - Client-side video transcription using Whisper AI
 * Uses Transformers.js to run Whisper directly in the browser
 * No server, no API keys, no microphone needed
 */

class Transcriber {
    constructor(options = {}) {
        this.options = {
            model: options.model || 'Xenova/whisper-tiny.en',
            language: options.language || 'en',
        };

        this.pipeline = null;
        this.isTranscribing = false;
        this.isCancelled = false;
        this.segments = [];
        this.listeners = {};
    }

    /**
     * Check if browser supports required features
     */
    isSupported() {
        return typeof AudioContext !== 'undefined' || typeof webkitAudioContext !== 'undefined';
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
     * Load the Whisper model
     */
    async loadModel(onProgress) {
        if (this.pipeline) return this.pipeline;

        const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');

        this.pipeline = await pipeline('automatic-speech-recognition', this.options.model, {
            progress_callback: (progress) => {
                if (progress.status === 'downloading' || progress.status === 'progress') {
                    const percent = progress.progress ? Math.round(progress.progress) : 0;
                    if (onProgress) onProgress({ status: 'loading', message: `Loading AI model... ${percent}%`, percent });
                    this.emit('loading', { percent, file: progress.file });
                }
            }
        });

        return this.pipeline;
    }

    /**
     * Extract audio from video/audio file and resample to 16kHz
     */
    async extractAudio(file, onProgress) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onprogress = (e) => {
                if (e.lengthComputable && onProgress) {
                    const percent = Math.round((e.loaded / e.total) * 100);
                    onProgress({ status: 'reading', message: `Reading file... ${percent}%`, percent });
                }
            };

            reader.onload = async (e) => {
                try {
                    if (onProgress) onProgress({ status: 'decoding', message: 'Decoding audio...', percent: 0 });

                    const arrayBuffer = e.target.result;
                    
                    // Use standard sample rate for decoding, then resample
                    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
                    const audioContext = new AudioContextClass();
                    
                    let audioBuffer;
                    try {
                        audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    } catch (decodeError) {
                        audioContext.close();
                        reject(new Error('Failed to decode audio. The file may not contain valid audio or the format is not supported.'));
                        return;
                    }

                    if (onProgress) onProgress({ status: 'processing', message: 'Processing audio...', percent: 50 });

                    // Resample to 16kHz mono (required by Whisper)
                    const targetSampleRate = 16000;
                    const audioData = this.resampleAudio(audioBuffer, targetSampleRate);

                    audioContext.close();
                    
                    console.log('Audio extracted:', audioData.length, 'samples at 16kHz');
                    resolve(audioData);
                } catch (error) {
                    console.error('Audio extraction error:', error);
                    reject(new Error('Failed to process audio: ' + error.message));
                }
            };

            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsArrayBuffer(file);
        });
    }

    /**
     * Resample audio to target sample rate and convert to mono
     */
    resampleAudio(audioBuffer, targetSampleRate) {
        const numChannels = audioBuffer.numberOfChannels;
        const sourceSampleRate = audioBuffer.sampleRate;
        const sourceLength = audioBuffer.length;
        
        // Calculate target length
        const targetLength = Math.round(sourceLength * targetSampleRate / sourceSampleRate);
        const result = new Float32Array(targetLength);
        
        // First, mix to mono
        const mono = new Float32Array(sourceLength);
        for (let channel = 0; channel < numChannels; channel++) {
            const channelData = audioBuffer.getChannelData(channel);
            for (let i = 0; i < sourceLength; i++) {
                mono[i] += channelData[i] / numChannels;
            }
        }
        
        // Then resample using linear interpolation
        const ratio = sourceSampleRate / targetSampleRate;
        for (let i = 0; i < targetLength; i++) {
            const srcIndex = i * ratio;
            const srcIndexFloor = Math.floor(srcIndex);
            const srcIndexCeil = Math.min(srcIndexFloor + 1, sourceLength - 1);
            const t = srcIndex - srcIndexFloor;
            result[i] = mono[srcIndexFloor] * (1 - t) + mono[srcIndexCeil] * t;
        }
        
        return result;
    }

    /**
     * Transcribe a video/audio file
     */
    async transcribe(file, options = {}) {
        if (!this.isSupported()) {
            throw new Error('AudioContext is not supported in this browser.');
        }

        const onProgress = options.onProgress || (() => { });
        const onPartialResult = options.onPartialResult || (() => { });

        this.isTranscribing = true;
        this.isCancelled = false;
        this.segments = [];

        try {
            this.emit('start');

            // Step 1: Load the model
            onProgress({ status: 'loading', message: 'Loading AI model...', percent: 0 });
            await this.loadModel((progress) => {
                onProgress(progress);
            });

            if (this.isCancelled) throw new Error('Cancelled');

            // Step 2: Extract audio from video
            onProgress({ status: 'extracting', message: 'Extracting audio from video...', percent: 0 });
            const audioData = await this.extractAudio(file, onProgress);

            if (this.isCancelled) throw new Error('Cancelled');

            // Step 3: Transcribe with Whisper
            onProgress({ status: 'transcribing', message: 'Transcribing audio (this may take a while)...', percent: 0 });
            onPartialResult('Processing audio with Whisper AI...');

            console.log('Starting Whisper transcription, audio length:', audioData.length / 16000, 'seconds');
            
            let result;
            try {
                result = await this.pipeline(audioData, {
                    chunk_length_s: 30,
                    stride_length_s: 5,
                    return_timestamps: true,
                });
            } catch (pipelineError) {
                console.error('Pipeline error:', pipelineError);
                throw new Error('Transcription failed: ' + pipelineError.message);
            }
            
            console.log('Whisper result:', result);

            if (this.isCancelled) throw new Error('Cancelled');

            // Process results
            let fullText = '';
            this.segments = [];

            if (result.chunks && result.chunks.length > 0) {
                this.segments = result.chunks.map(chunk => ({
                    text: chunk.text.trim(),
                    startTime: chunk.timestamp[0] || 0,
                    endTime: chunk.timestamp[1] || 0,
                    confidence: 0.9
                }));
                fullText = this.segments.map(s => s.text).join(' ');
            } else {
                fullText = result.text || '';
            }

            onProgress({ status: 'complete', message: 'Transcription complete!', percent: 100 });

            const finalResult = {
                text: fullText.trim(),
                segments: this.segments,
                duration: this.segments.length > 0 ? this.segments[this.segments.length - 1].endTime : 0,
                language: this.options.language
            };

            this.emit('end', finalResult);
            this.isTranscribing = false;

            return finalResult;

        } catch (error) {
            this.isTranscribing = false;
            if (error.message === 'Cancelled') {
                this.emit('cancel');
                throw error;
            }
            this.emit('error', { error });
            throw error;
        }
    }

    /**
     * Cancel ongoing transcription
     */
    cancel() {
        this.isCancelled = true;
        this.isTranscribing = false;
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
        if (seconds === null || seconds === undefined || isNaN(seconds)) {
            return '00:00';
        }
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }
}

// ===== UI Code (only runs on main page) =====

document.addEventListener('DOMContentLoaded', () => {
    console.log('Transcriptify: DOM loaded, initializing...');

    // Check if we're on the main page
    const dropzone = document.getElementById('dropzone');
    if (!dropzone) {
        console.log('Transcriptify: Not on main page (no dropzone found)');
        return;
    }

    console.log('Transcriptify: Found dropzone, setting up event handlers');

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
        if (files.length > 0) {
            const file = files[0];
            // Check MIME type or file extension
            const isVideo = file.type.startsWith('video/') ||
                file.type.startsWith('audio/') ||
                /\.(mp4|webm|mov|avi|mkv|mp3|wav|m4a|ogg)$/i.test(file.name);
            if (isVideo) {
                handleFile(file);
            } else {
                console.log('Rejected file:', file.name, file.type);
            }
        }
    });

    dropzone.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        console.log('File input change:', e.target.files);
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        console.log('handleFile called with:', file.name, file.type, file.size);
        currentFile = file;

        // Show video preview
        const videoUrl = URL.createObjectURL(file);
        videoPlayer.src = videoUrl;

        // Show processing section
        processingSection.classList.remove('hidden');
        resultSection.classList.add('hidden');

        // Reset state
        statusEl.textContent = `Loaded: ${file.name}`;
        statusEl.style.color = '';
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

        statusEl.textContent = 'Initializing...';
        statusEl.style.color = '';
        transcriptEl.innerHTML = '<span style="color: var(--text-muted);">Preparing to transcribe...</span>';

        try {
            transcriptionResult = await transcriber.transcribe(currentFile, {
                onProgress: (progress) => {
                    statusEl.textContent = progress.message || progress.status;
                    if (progress.percent !== undefined) {
                        progressEl.style.width = `${progress.percent}%`;
                        progressText.textContent = `${progress.percent}%`;
                    }

                    // Update transcript area with status
                    if (progress.status === 'loading') {
                        transcriptEl.innerHTML = `<span style="color: var(--text-muted);">Loading Whisper AI model... This may take a moment on first use.</span>`;
                    } else if (progress.status === 'extracting' || progress.status === 'decoding') {
                        transcriptEl.innerHTML = `<span style="color: var(--text-muted);">Extracting audio from video...</span>`;
                    } else if (progress.status === 'transcribing') {
                        transcriptEl.innerHTML = `<span style="color: var(--text-muted);">Transcribing with Whisper AI... This runs entirely in your browser.</span>`;
                    }
                },
                onPartialResult: (text) => {
                    if (text.trim()) {
                        transcriptEl.innerHTML = `<span style="color: var(--text-muted);">${text}</span>`;
                    }
                }
            });

            statusEl.textContent = 'Transcription complete!';
            cancelBtn.classList.add('hidden');
            progressEl.style.width = '100%';
            progressText.textContent = '100%';

            // Display final result with timestamps
            displayTranscript(transcriptionResult);

        } catch (error) {
            console.error('Transcription error:', error);
            if (error.message !== 'Cancelled') {
                statusEl.textContent = `Error: ${error.message}`;
                statusEl.style.color = '#ff6b6b';
            } else {
                statusEl.textContent = 'Transcription cancelled';
            }
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
