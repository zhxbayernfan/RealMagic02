import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image

class LocalVLM:
    def __init__(self, model_id="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda", load_in_4bit=True):
        self.device = device
        self.model_id = model_id
        
        print(f"Loading LocalVLM: {model_id} (4bit={load_in_4bit})...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id, 
                quantization_config=quantization_config, 
                device_map="auto"
            )
        else:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                device_map="auto"
            )
        print("LocalVLM loaded.")

    def generate_content(self, image_input, prompt):
        """
        Generic generation method.
        image_input: PIL Image or None (if text-only, though LLaVA usually expects image)
        prompt: Text prompt
        """
        # Create conversation format
        if image_input is not None:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(text=text_prompt, images=image_input, return_tensors="pt").to(self.device)
        else:
            # Fallback for text-only (if model supports it, otherwise we might need a dummy image)
            # LLaVA 1.6 usually expects an image. If text-only is needed for relations, 
            # we might want to feed the object images again or use a blank image.
            # For safety with LLaVA, let's assume we always pass an image or handle it upstream.
            # But if we really need text-only, we can try omitting image token.
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(text=text_prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=200)
            
        decoded_output = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extract the assistant's response (remove the prompt part)
        # The prompt usually ends with specific tokens, but decode might keep them.
        # A simple heuristic: split by [/INST] or similar if present, or just take the new tokens.
        # However, decode(output[0]) includes input. 
        # We can slice output tokens.
        generated_tokens = output[0][inputs['input_ids'].shape[1]:]
        response = self.processor.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()
