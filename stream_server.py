import sounddevice as sd
import torch
import numpy as np
from omegaconf import OmegaConf
from chattts_plus.pipelines.chattts_plus_pipeline import ChatTTSPlusPipeline
from chattts_plus.commons.utils import InferCodeParams, RefineTextParams
import threading
import zerorpc
from collections import deque
import re
import time

class RingBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.lock = threading.Lock()

    def put(self, data):
        with self.lock:
            self.buffer.append(data)

    def get(self):
        with self.lock:
            if self.buffer:
                return self.buffer.popleft()
            return None

RING_BUFFER_SIZE = 40  # Adjust this based on your needs
audio_buffer = RingBuffer(RING_BUFFER_SIZE)
input_text_buffer = RingBuffer(RING_BUFFER_SIZE)
audio_finished_event = threading.Event()
AUDIO_PLAYBACK_TIMEOUT = 1.0
last_audio_time = time.time()
emergency_stop_flag = threading.Event()
thread_exit_event = threading.Event()
playing = False
infer_start_time = None

def replace_arabic_numbers_with_chinese(text):
    translation_table = str.maketrans({
        '0': '零',
        '1': '一',
        '2': '二',
        '3': '三',
        '4': '四',
        '5': '五',
        '6': '六',
        '7': '七',
        '8': '八',
        '9': '九'
    }) 
    replaced_text = re.sub(r'\d+', lambda x: x.group().translate(translation_table), text)
    return replaced_text

def monitor_audio_state():
    """Thread to monitor and update audio_playing_state based on timeout."""
    global last_audio_time, audio_finished_event, thread_exit_event, playing, infer_start_time
    was_playing = False  # Track the previous state
    while not thread_exit_event.is_set():
        time.sleep(0.01)  # Check periodically

        is_playing = time.time() - last_audio_time <= AUDIO_PLAYBACK_TIMEOUT

        if not was_playing and is_playing:
            playback_start_time = time.time()
            print("\nAudio started playing...")
            if infer_start_time is not None:
                latency = playback_start_time - infer_start_time
                print(f"\n=== Latency from infer to playback: {latency:.3f} seconds ===\n")
                infer_start_time = None 
            audio_finished_event.clear()
            playing = True
        elif was_playing and not is_playing:
            print("\nAudio finished playing...")
            audio_finished_event.set()
            playing = False

        was_playing = is_playing

def audio_playback_thread(audio_buffer, samplerate=24000, blocksize=12000):
    global emergency_stop_flag, thread_exit_event
    def callback(outdata, frames, time_info, status):
        global last_audio_time
        if emergency_stop_flag.is_set():
            outdata[:] = b'\x00' * len(outdata)
            return
        chunk = audio_buffer.get()
        if chunk is not None:    
            chunk_length = min(len(chunk), len(outdata))
            outdata[:chunk_length] = chunk[:chunk_length]
            if len(outdata) > chunk_length:
                outdata[chunk_length:] = b'\x00' * (len(outdata) - chunk_length)
            last_audio_time = time.time()
        else:
            outdata[:] = b'\x00' * len(outdata)
    while not thread_exit_event.is_set():
        try:
            with sd.RawOutputStream(
                samplerate=samplerate,
                blocksize=blocksize,
                channels=1, 
                latency=0.001,
                dtype='float32',
                callback=callback
            ):
                print("Audio playback Thread started.")
                while True:
                    sd.sleep(1000)  # Keep the stream alive
        except Exception as e:
            print("Stream aborted, restarting...",e)
    print("Audio playback thread exited.")
        


class TTSServer:
    def __init__(self):
        self.emergency_stopped = False
        infer_cfg_path = "configs/infer/chattts_plus_trt.yaml"
        infer_cfg = OmegaConf.load(infer_cfg_path)
        self.pipeline = ChatTTSPlusPipeline(infer_cfg, device=torch.device("cuda"))
        self.speaker_emb_path = "spk0.pt"
        self.params_infer_code = InferCodeParams(
            prompt="[speed_5]",
            temperature=.0003,
            max_new_token=2048,
            top_P=0.7,
            top_K=20,
            pass_first_n_batches=0
        )
        self.params_refine_text = RefineTextParams(
            prompt='[oral_2][laugh_0][break_4]',
            top_P=0.7,
            top_K=20,
            temperature=0.3,
            max_new_token=384
        )
        self.playback_thread = threading.Thread(
            target=audio_playback_thread,
            args=(audio_buffer,),
            kwargs={'samplerate': 24000, 'blocksize': 12000}
        )
        self.playback_thread.start()

        self.state_monitor_thread = threading.Thread(target=monitor_audio_state)
        self.state_monitor_thread.start()

        self.text_processing_thread = threading.Thread(target=self.text_processing_thread)
        self.text_processing_thread.start()

    def text_processing_thread(self):
        global thread_exit_event
        while not thread_exit_event.is_set():
            text = input_text_buffer.get()
            if text is not None:
                sentences = [text]
                print("\n Sentences:",text)
                stream_gen = self.pipeline.infer(
                    sentences,
                    params_refine_text=self.params_refine_text,
                    params_infer_code=self.params_infer_code,
                    use_decoder=True,
                    stream=True,  # Enable streaming
                    skip_refine_text=False,
                    do_text_normalization=False,
                    do_homophone_replacement=False,
                    do_text_optimization=False,
                    speaker_emb_path=self.speaker_emb_path
                )

                for chunk in stream_gen:
                    if chunk is not None:
                        chunk_bytes = np.array(chunk[0].cpu(), dtype=np.float32).tobytes()
                        audio_buffer.put(chunk_bytes)
                    else:
                        break
            else:
                time.sleep(0.01)  # Sleep briefly if no text is available

    def infer(self, payload, ctx={'peer': '127.0.0.1'}):
        global audio_finished_event, infer_start_time
        if self.emergency_stopped:
            print("Server is in emergency stop state. Ignoring request.")
            return
        text = payload.get("text", "")
        if not text:
            return
        caller_ip = ctx.get('peer', 'unknown')
        audio_finished_event.clear()
        text = replace_arabic_numbers_with_chinese(text)
        input_text_buffer.put(text)  # Put the text into the input ring buffer
        infer_start_time = time.time()

    def emergency_stop(self):
        global emergency_stop_flag
        emergency_stop_flag.set()
        audio_buffer.lock.acquire()
        audio_buffer.buffer.clear()
        audio_buffer.lock.release()
        self.emergency_stopped = True
        print("Emergency stop initiated. Clearing buffer and aborting audio playback.")

    def reset_emergency_stop(self):
        global emergency_stop_flag
        emergency_stop_flag.clear()
        self.emergency_stopped = False
        print("Emergency reset completed. Server is ready to accept new requests.")

    def check_audio_finished(self):
        global audio_finished_event
        print("\nWaiting for audio to finish...")
        audio_finished_event.wait()  # Wait for playback to complete
    
    def get_playback_status(self):
        global playing
        if playing:
            return 'playing'
        else:
            return 'finished'

    def __del__(self):
        global emergency_stop_flag,thread_exit_event
        thread_exit_event.set()
        emergency_stop_flag.set()
        self.playback_thread.join()
        self.state_monitor_thread.join()
        self.text_processing_thread.join()  # Join the text processing thread

if __name__ == "__main__":
    server = zerorpc.Server(TTSServer())
    server.bind("tcp://0.0.0.0:4242")
    print("TTS Server is running on port 4242...")
    server.run()