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

RING_BUFFER_SIZE = 200  # Adjust this based on your needs
audio_buffer = RingBuffer(RING_BUFFER_SIZE)

def audio_playback_thread(audio_buffer, samplerate=24000, blocksize=12000):

    def callback(outdata, frames, time, status):
            chunk = audio_buffer.get()
            if chunk is not None:
                chunk_length = min(len(chunk), len(outdata))
                outdata[:chunk_length] = chunk[:chunk_length]
                if len(outdata) > chunk_length:
                    outdata[chunk_length:] = b'\x00' * (len(outdata) - chunk_length)
            else:
                outdata[:] = b'\x00' * len(outdata)
    with sd.RawOutputStream(
        samplerate=samplerate,
        blocksize=blocksize,
        channels=1,
        dtype='float32',
        callback=callback
    ):
        print("Audio playback started.")
        while True:
            sd.sleep(1000)  # Keep the stream alive

def stream_tts_example(pipeline, sentences, params_refine_text, params_infer_code, speaker_emb_path, audio_buffer):
    print("\nsentences : ", sentences)
    stream_gen = pipeline.infer(
        sentences,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
        use_decoder=True,
        stream=True,  # Enable streaming
        skip_refine_text=True,
        do_text_normalization=False,
        do_homophone_replacement=False,
        do_text_optimization=False,
        speaker_emb_path=speaker_emb_path
    )

    print("Starting audio streaming...")
    for chunk in stream_gen:
        if chunk is not None:
            chunk_bytes = np.array(chunk[0].cpu(), dtype=np.float32).tobytes()
            audio_buffer.put(chunk_bytes)
        else:
            break
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

# TTS Server Class
class TTSServer:
    def __init__(self):
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

    def infer(self, payload):
        text = payload.get("text", "")
        if not text:
            return
        text = replace_arabic_numbers_with_chinese(text)
        sentences = [text]
        stream_tts_example(
            self.pipeline,
            sentences,
            self.params_refine_text,
            self.params_infer_code,
            self.speaker_emb_path,
            audio_buffer
        )

    def __del__(self):
        self.playback_thread.join()

if __name__ == "__main__":
    server = zerorpc.Server(TTSServer())
    server.bind("tcp://0.0.0.0:4242")
    print("TTS Server is running on port 4242...")
    server.run()
