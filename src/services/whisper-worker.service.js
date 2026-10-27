const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { spawn } = require('child_process');
const logger = require('../core/logger').createServiceLogger('WHISPER-WORKER');

class WhisperWorkerService {
  constructor() {
    this.process = null;
    this.pythonPath = null;
    this.scriptPath = null;
    this.pending = new Map();
    this.sequence = 0;
    this.idleTimer = null;
    this.idleUnloadMs = 60000;
    this.readyInfo = null;
    this.closing = false;
    // While a recording session is live the model must stay resident; an idle
    // unload mid-session forces a multi-second reload on the next utterance.
    this.holdLoaded = false;
    this.lastResult = null;
    this.latencyEmaMs = null;
  }

  configure({ pythonPath, scriptPath, idleUnloadMs = 60000 }) {
    const changed = this.pythonPath !== pythonPath || this.scriptPath !== scriptPath;
    this.pythonPath = pythonPath;
    this.scriptPath = scriptPath;
    this.idleUnloadMs = idleUnloadMs;
    if (changed) {
      this.close();
    }
  }

  isConfigured() {
    return Boolean(
      this.pythonPath &&
      this.scriptPath &&
      fs.existsSync(this.pythonPath) &&
      fs.existsSync(this.scriptPath)
    );
  }

  /**
   * Transcribe either a file (audioPath) or raw 16 kHz mono PCM16 (options.pcm).
   * Passing PCM inline skips the temp WAV write/read round trip.
   */
  async transcribe(audioPath, options = {}) {
    if (!this.isConfigured()) {
      throw new Error('Persistent Whisper worker is not configured');
    }
    this._clearIdleTimer();
    const payload = {
      action: 'transcribe',
      model: options.model || 'small',
      language: options.language || 'auto',
      model_dir: options.modelDir || null,
      device: options.device || 'auto',
      backend: options.backend || 'auto',
      prompt: options.prompt || null,
      beam_size: options.beamSize || 1
    };
    if (options.pcm) {
      payload.audio_b64 = Buffer.from(options.pcm).toString('base64');
    } else {
      payload.audio_path = audioPath;
    }
    const startedAt = Date.now();
    try {
      const result = await this._request(payload);
      this.lastResult = result;
      const latency = Date.now() - startedAt;
      this.latencyEmaMs = this.latencyEmaMs == null ? latency : this.latencyEmaMs * 0.7 + latency * 0.3;
      return result;
    } finally {
      this._scheduleIdleUnload();
    }
  }

  /**
   * True when transcriptions come back fast enough for live partials. A slow
   * partial would hold the single worker and delay the real transcript.
   */
  isFastBackend(maxLatencyMs = 900) {
    return this.latencyEmaMs != null && this.latencyEmaMs <= maxLatencyMs;
  }

  isBusy() {
    return this.pending.size > 0;
  }

  setHoldLoaded(hold) {
    this.holdLoaded = Boolean(hold);
    if (this.holdLoaded) {
      this._clearIdleTimer();
    } else if (this.process) {
      this._scheduleIdleUnload();
    }
  }

  async warmup(options = {}) {
    if (!this.isConfigured()) {
      throw new Error('Persistent Whisper worker is not configured');
    }
    this._clearIdleTimer();
    const result = await this._request({
      action: 'warmup',
      model: options.model || 'small',
      model_dir: options.modelDir || null,
      device: options.device || 'auto',
      backend: options.backend || 'auto'
    });
    this.lastResult = result;
    this._scheduleIdleUnload();
    return result;
  }

  releaseWhenIdle() {
    if (this.process) {
      this._scheduleIdleUnload();
    }
  }

  _ensureProcess() {
    if (this.process && !this.process.killed) {
      return;
    }

    this.readyInfo = null;
    this.closing = false;
    const child = spawn(this.pythonPath, ['-u', this.scriptPath], {
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true
    });
    this.process = child;

    const lines = readline.createInterface({ input: child.stdout });
    lines.on('line', (line) => this._handleLine(line));

    child.stderr.on('data', (chunk) => {
      const message = chunk.toString().trim();
      if (message) {
        logger.debug('Worker stderr', { message: message.substring(0, 1000) });
      }
    });

    child.on('error', (error) => {
      logger.error('Whisper worker process error', { error: error.message });
      this._rejectAll(error);
    });

    child.on('close', (code) => {
      const error = new Error(`Whisper worker exited with code ${code}`);
      if (!this.closing && code !== 0) {
        logger.error(error.message);
      }
      this.process = null;
      this.readyInfo = null;
      this.lastResult = null;
      this.latencyEmaMs = null;
      if (!this.closing) {
        this._rejectAll(error);
      }
      this.closing = false;
    });
  }

  _handleLine(line) {
    let message;
    try {
      message = JSON.parse(line);
    } catch (error) {
      logger.warn('Ignoring non-JSON worker output', { line: line.substring(0, 500) });
      return;
    }

    if (message.event === 'ready') {
      this.readyInfo = message;
      logger.info('Persistent Whisper worker ready', message);
      return;
    }

    const pending = this.pending.get(message.id);
    if (!pending) {
      return;
    }
    this.pending.delete(message.id);
    clearTimeout(pending.timer);

    if (message.ok) {
      pending.resolve(message);
    } else {
      const error = new Error(message.error || 'Whisper worker request failed');
      error.workerTraceback = message.traceback;
      pending.reject(error);
    }
  }

  _request(payload, timeoutMs = 180000) {
    this._ensureProcess();
    const id = ++this.sequence;
    const request = { ...payload, id };

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Whisper worker timed out after ${timeoutMs}ms`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });

      this.process.stdin.write(`${JSON.stringify(request)}\n`, (error) => {
        if (error) {
          clearTimeout(timer);
          this.pending.delete(id);
          reject(error);
        }
      });
    });
  }

  _scheduleIdleUnload() {
    this._clearIdleTimer();
    if (this.holdLoaded) {
      return;
    }
    this.idleTimer = setTimeout(() => {
      if (!this.process || this.pending.size > 0 || this.holdLoaded) {
        return;
      }
      this._request({ action: 'unload' }, 30000)
        .then(() => logger.info('Whisper model unloaded after idle timeout'))
        .catch((error) => logger.warn('Could not unload idle Whisper model', { error: error.message }));
    }, this.idleUnloadMs);
  }

  _clearIdleTimer() {
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
  }

  _rejectAll(error) {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
  }

  close() {
    this._clearIdleTimer();
    if (this.process && !this.process.killed) {
      try {
        this.closing = true;
        this.process.kill();
      } catch (_) {
        // Ignore shutdown races.
      }
    }
    this.process = null;
    this.readyInfo = null;
    this._rejectAll(new Error('Whisper worker closed'));
  }
}

module.exports = WhisperWorkerService;
