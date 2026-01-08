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
     * Extract audio from video/audio file
     */
    async extractAudio(file, onProgress) {
        return new Promise((resolve, reject) => {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            const audioContext = new AudioContextClass({ sampleRate: 16000 });

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
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

                    // Convert to mono Float32Array at 16kHz (required by Whisper)
                    const audioData = this.convertToMono(audioBuffer);

                    audioContext.close();
                    resolve(audioData);
                } catch (error) {
                    audioContext.close();
                    reject(new Error('Failed to decode audio. Make sure the file contains valid audio.'));
                }
            };

            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsArrayBuffer(file);
        });
    }

    /**
     * Convert AudioBuffer to mono Float32Array
     */
    convertToMono(audioBuffer) {
        const numChannels = audioBuffer.numberOfChannels;
        const length = audioBuffer.length;
        const result = new Float32Array(length);

        if (numChannels === 1) {
            return audioBuffer.getChannelData(0);
        }

        // Mix all channels to mono
        for (let channel = 0; channel < numChannels; channel++) {
            const channelData = audioBuffer.getChannelData(channel);
            for (let i = 0; i < length; i++) {
                result[i] += channelData[i] / numChannels;
            }
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
            onProgress({ status: 'transcribing', message: 'Transcribing audio...', percent: 0 });
            onPartialResult('Processing audio with Whisper AI...');

            const result = await this.pipeline(audioData, {
                chunk_length_s: 30,
                stride_length_s: 5,
                return_timestamps: true,
                language: this.options.language === 'en' ? null : this.options.language,
            });

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
        if (files.length > 0 && (files[0].type.startsWith('video/') || files[0].type.startsWith('audio/'))) {
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
