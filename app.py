from transformers import AutoProcessor, AutoModel
import numpy as np
import io
import base64
import soundfile as sf


class InferlessPythonModel:
    def initialize(self):
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.model = AutoModel.from_pretrained("suno/bark")
    
    def infer(self, inputs):
        prompt = inputs["prompt"]
        inputs = self.processor(
            text=[prompt],
            return_tensors="pt",
        )
        speech_values = self.model.generate(**inputs, do_sample=True)
        audio_numpy = speech_values.cpu().numpy().squeeze()

        buffer = io.BytesIO()
        sf.write(buffer, audio_numpy, self.model.generation_config.sample_rate, format='WAV')
        buffer.seek(0)

        base64_audio = base64.b64encode(buffer.read()).decode('utf-8')
        return {"generated_audio_base64": base64_audio}

    def finalize(self,args):
        self.pipe = None
