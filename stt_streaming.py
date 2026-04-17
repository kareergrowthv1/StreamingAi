"""
Live STT via AssemblyAI Streaming (WebSocket) – same pattern as KareerGrowth/AiService/stt_streaming.
Uses the official AssemblyAI Python SDK so we do not call the token API directly (avoids 422 from invalid params).
Receives PCM audio, resamples to 16kHz if needed, streams to AssemblyAI, forwards transcripts via callback.
"""
import asyncio
import logging
import queue
import threading

logger = logging.getLogger(__name__)

# Optional: certifi for SSL (SDK may use requests/websockets under the hood)
def _install_ssl_certifi():
    try:
        import os
        import ssl as _ssl
        import certifi
        _ctx = _ssl.create_default_context()
        _ctx.load_verify_locations(certifi.where())
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except Exception as e:
        logging.getLogger(__name__).warning("Could not install certifi SSL: %s", e)


_install_ssl_certifi()

SAMPLE_RATE_TARGET = 16000
SAMPLE_RATE_CLIENT = 16000
BYTES_PER_SAMPLE = 2


def _resample_to_16k(pcm_bytes: bytes, from_rate: int) -> bytes:
    """Resample PCM16 mono from from_rate to 16000 Hz."""
    if from_rate == SAMPLE_RATE_TARGET:
        return pcm_bytes
    try:
        import numpy as np
        from scipy import signal
    except ImportError:
        return pcm_bytes
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    n_out = int(len(arr) * SAMPLE_RATE_TARGET / from_rate)
    resampled = signal.resample(arr.astype(np.float64), n_out)
    return resampled.astype(np.int16).tobytes()


def _audio_generator(audio_queue: queue.Queue, done: threading.Event, from_rate: int = SAMPLE_RATE_CLIENT):
    """Yields PCM16 16kHz mono bytes for AssemblyAI from queue until done."""
    while not done.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
            if chunk is None:
                break
            data = _resample_to_16k(chunk, from_rate)
            if data:
                yield data
        except queue.Empty:
            continue
        except Exception as e:
            logger.warning("STT audio generator error: %s", e)
            break


def run_assemblyai_stream(
    api_key: str,
    audio_queue: queue.Queue,
    done: threading.Event,
    send_callback,
    loop: asyncio.AbstractEventLoop,
    client_sample_rate: int = SAMPLE_RATE_CLIENT,
):
    """
    Run AssemblyAI streaming in a thread. Uses official SDK (no token API call).
    send_callback(msg_dict) must return a coroutine (e.g. websocket.send_json(msg)); it is run on loop.
    """
    try:
        import assemblyai as aai
        from assemblyai.streaming.v3 import (
            StreamingClient,
            StreamingClientOptions,
            StreamingParameters,
            StreamingEvents,
            TurnEvent,
            BeginEvent,
            TerminationEvent,
            StreamingError,
        )
    except ImportError as e:
        logger.error("AssemblyAI SDK not available: %s", e)
        return

    def send_to_client(obj: dict):
        try:
            coro = send_callback(obj)
            if coro is not None:
                asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as e:
            logger.warning("STT could not schedule send: %s", e)

    def on_begin(_client, event: BeginEvent):
        logger.info("AssemblyAI session started: %s", getattr(event, "id", ""))

    def on_turn(_client, event: TurnEvent):
        text = (event.transcript or "").strip()
        if not text:
            return
        end_of_turn = getattr(event, "end_of_turn", True)
        # Match Streaming AI WebSocket format: type "transcript", text, is_final
        msg = {"type": "transcript", "text": text, "is_final": bool(end_of_turn)}
        send_to_client(msg)

    def on_terminated(_client, event: TerminationEvent):
        logger.info("AssemblyAI session terminated, audio_duration_seconds=%s", getattr(event, "audio_duration_seconds", ""))

    def on_error(_client, err: StreamingError):
        logger.error("AssemblyAI streaming error: %s", err)
        send_to_client({"type": "transcript", "error": str(err)})

    client = StreamingClient(
        StreamingClientOptions(api_key=api_key, api_host="streaming.assemblyai.com")
    )
    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    try:
        client.connect(StreamingParameters(sample_rate=SAMPLE_RATE_TARGET, format_turns=True, speech_model="universal-streaming-english"))
        gen = _audio_generator(audio_queue, done, client_sample_rate)
        client.stream(gen)
    except Exception as e:
        logger.exception("AssemblyAI stream failed: %s", e)
        send_to_client({"type": "transcript", "error": str(e)})
    finally:
        try:
            client.disconnect(terminate=True)
        except Exception:
            pass


class AssemblyAIStreamRunner:
    """
    Runs AssemblyAI streaming in a background thread. Main loop feeds audio via put_audio(), stop() to end.
    """

    def __init__(self, api_key: str, loop: asyncio.AbstractEventLoop, send_json_coro_fn):
        """
        send_json_coro_fn(msg_dict) must return a coroutine that sends msg_dict (e.g. websocket.send_json(msg)).
        """
        self.api_key = api_key
        self.loop = loop
        self.send_json_coro_fn = send_json_coro_fn
        self.audio_queue = queue.Queue()
        self.done = threading.Event()
        self.thread = None

    def start(self, client_sample_rate: int = SAMPLE_RATE_CLIENT):
        # Pass a callback that returns a coroutine so run_assemblyai_stream can schedule it on the loop
        send_cb = lambda msg: self.send_json_coro_fn(msg)
        self.done.clear()
        self.thread = threading.Thread(
            target=run_assemblyai_stream,
            args=(self.api_key, self.audio_queue, self.done, send_cb, self.loop, client_sample_rate),
            daemon=True,
        )
        self.thread.start()

    def put_audio(self, pcm_bytes: bytes):
        if not self.done.is_set():
            self.audio_queue.put(pcm_bytes)

    def stop(self, join_timeout: float = 5.0):
        self.done.set()
        self.audio_queue.put(None)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=join_timeout)
        self.thread = None
