from PIL import Image

import soundfile as sf
import gradio as gr
import numpy as np
import traceback
import librosa
import torch
import os

from tempfile import NamedTemporaryFile

MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
USE_FLASH_ATTENTION = False

def is_flash_attn_available():
    try:
        import flash_attn
        return True
    except ImportError:
        return False

def ensure_qwen_omni_utils():
    """Ensure the qwen_omni_utils module is available."""
    return QWEN_UTILS_AVAILABLE

try:
    try:
        from qwen_omni_utils.v2_5 import vision_process
        from qwen_omni_utils.v2_5 import audio_process

        vision_funcs = dir(vision_process)
        audio_funcs = dir(audio_process)
        print(f"Available vision functions: {vision_funcs}")
        print(f"Available audio functions: {audio_funcs}")

        QWEN_UTILS_V2_5_AVAILABLE = True
    except ImportError:
        QWEN_UTILS_V2_5_AVAILABLE = False
    print(f"Official qwen-omni-utils is available. V2.5 modules: {QWEN_UTILS_V2_5_AVAILABLE}")
    QWEN_UTILS_AVAILABLE = True
except ImportError:
    print("Official qwen-omni-utils is not available.")
    print("For better multi-modal support, install it with: pip install qwen-omni-utils[decord]")
    QWEN_UTILS_AVAILABLE = False
    QWEN_UTILS_V2_5_AVAILABLE = False

def process_image_with_vision_module(image):
    """Process image using the vision_process module if available."""
    if not QWEN_UTILS_V2_5_AVAILABLE:
        return image
    
    try:
        if hasattr(image, 'size'):
            image = resize_image_if_needed(image)
        if 'process_vision_info' in vision_funcs:
            return vision_process.process_vision_info(image)
        elif 'smart_resize' in vision_funcs:
            return vision_process.smart_resize(image)
        elif 'extract_vision_info' in vision_funcs:
            return vision_process.extract_vision_info(image)
        else:
            print("No suitable vision processing function found")
            return image
    except Exception as e:
        print(f"Error processing image with vision module: {e}")
        return image

def ensure_audio_tensor_format(audio_data):
    """Convert audio data to the correct format for the model."""
    try:
        if audio_data is None:
            return None
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data)

            if torch.cuda.is_available():
                audio_tensor = audio_tensor.to(dtype=torch.float16)
            
            print(f"Converted audio numpy array to tensor with dtype {audio_tensor.dtype}")
            return audio_tensor
        
        elif isinstance(audio_data, torch.Tensor):
            if torch.cuda.is_available():
                audio_tensor = audio_data.to(dtype=torch.float16)
            else:
                audio_tensor = audio_data
                
            print(f"Ensured audio tensor has dtype {audio_tensor.dtype}")
            return audio_tensor
            
        return audio_data
    
    except Exception as e:
        print(f"Error ensuring audio tensor format: {e}")
        return audio_data

def load_audio_file(audio_path, sample_rate=16000):
    """Load audio file using librosa and return numpy array."""
    try:
        audio_data, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        audio_data = audio_data.astype(np.float32)

        return audio_data
    except Exception as e:
        print(f"Error loading audio with librosa: {e}")
        return None

def process_audio_with_audio_module(audio_path, use_audio_in_video=False):
    """Process audio using the audio_process module if available."""
    try:
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return None

        if QWEN_UTILS_AVAILABLE and QWEN_UTILS_V2_5_AVAILABLE:
            print(f"Processing audio with qwen_omni_utils.v2_5.audio_process module")
            try:
                audio_data, _ = librosa.load(audio_path, sr=16000, mono=True)
                print(f"Loaded audio data with librosa: shape={audio_data.shape}, dtype={audio_data.dtype}")
                audio_data = audio_data.astype(np.float32)
                
                from qwen_omni_utils.v2_5 import audio_process
                if hasattr(audio_process, 'process_audio_tensor'):
                    processed_audio = audio_process.process_audio_tensor(audio_data)
                elif hasattr(audio_process, 'process_audio'):
                    processed_audio = audio_process.process_audio(audio_data)
                else:
                    print("No suitable audio processing function found in qwen_omni_utils.v2_5.audio_process")
                    return audio_data
                
                print(f"Processed audio with qwen_omni_utils: {type(processed_audio)}")
                return processed_audio
            except Exception as e:
                print(f"Error processing audio with qwen_omni_utils: {e}")
                traceback.print_exc()

        print(f"Falling back to librosa for audio processing")
        audio_data, _ = librosa.load(audio_path, sr=16000, mono=True)
        return audio_data
        
    except Exception as e:
        print(f"Error processing audio with audio module: {e}")
        traceback.print_exc()
        return None

def resize_image_if_needed(img, max_width=1024, max_height=1024):
    """Resize image if it's too large to avoid memory issues."""
    if img.width > max_width or img.height > max_height:
        ratio = min(max_width / img.width, max_height / img.height)
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        print(f"Resizing image from {img.width}x{img.height} to {new_width}x{new_height}")
        return img.resize((new_width, new_height), Image.LANCZOS)
    return img

class MultiModalUI:
    """Class to handle the UI and model interactions."""
    
    def __init__(self):
        """Initialize the UI with model and processor."""
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.use_multimodal = False
        self.model_loaded = False
        self.conversation_history = []
        self.temperature = 0.7
        self.max_new_tokens = 512
        self.load_model()
    
    def reset_state(self):
        """Reset any internal state variables."""
        print("Internal state reset")
        self.conversation_history = []
    
    def load_model(self):
        """Load the Qwen 2.5 Omni model and processor."""
        print("Loading Qwen 2.5 Omni model. This may take a few minutes...")
        
        qwen_utils_available = ensure_qwen_omni_utils()
        
        try:
            from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
            
            if qwen_utils_available:
                from qwen_omni_utils import process_mm_info
                self.process_mm_info = process_mm_info
            else:
                raise ImportError("qwen_omni_utils module not available")
            
            self.model_loaded = True
            self.use_multimodal = True
            
            model_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else "auto",
                "device_map": "auto",
                "trust_remote_code": True
            }
            
            print("Flash Attention is disabled via environment variable. Model will choose an appropriate attention mechanism.")
            print(f"Loading processor from {MODEL_NAME}...")
            self.processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
            
            print(f"Loading model from {MODEL_NAME}...")
            os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
            self.model = Qwen2_5OmniModel.from_pretrained(MODEL_NAME, **model_kwargs)
            
            print("Qwen 2.5 Omni model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading Qwen 2.5 Omni model: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to basic text model...")
            
            self.model_loaded = True
            self.use_multimodal = False
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            print("Fallback text model loaded successfully!")
    
    def generate_response(self, user_message, direct_image=None, direct_audio=None, history=None):
        """Generate a response from the model based on user input and history."""
        try:
            if history is None:
                history = []
                
            print(f"Debug before try-catch: conversation={len(history)}, direct_image={direct_image}")
            
            if len(history) == 0:
                print("Starting with a fresh conversation")
                conversation = [
                    {"role": "system", "content": SYSTEM_PROMPT}
                ]
            else:
                conversation = []
                for h_msg in history:
                    if not isinstance(h_msg, tuple) or len(h_msg) != 2:
                        continue
                    
                    user_part, assistant_part = h_msg
                    if user_part is None or assistant_part is None:
                        continue
                    
                    if isinstance(user_part, str) and user_part.strip():
                        conversation.append({
                            "role": "user", 
                            "content": [{"type": "text", "text": user_part.strip()}]
                        })
                    
                    if isinstance(assistant_part, str) and assistant_part.strip():
                        conversation.append({
                            "role": "assistant", 
                            "content": assistant_part.strip()
                        })
                
                if not any(msg.get("role") == "system" for msg in conversation):
                    conversation.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
            
            user_content = []
            
            if direct_image:
                print("Processing image with the correct modules")
                try:
                    if isinstance(direct_image, str) and os.path.exists(direct_image):
                        try:
                            pil_image = Image.open(direct_image)
                            pil_image = resize_image_if_needed(pil_image)
                            image_data = process_image_with_vision_module(pil_image)

                            print(f"Processed image with vision_process: {type(image_data)}")
                        except Exception as e:
                            print(f"Error processing image: {e}")
                            pil_image = Image.open(direct_image)
                            image_data = resize_image_if_needed(pil_image)
                            
                            print(f"Using resized PIL image: {image_data.size}")
                        
                        user_content.append({
                            "type": "image",
                            "image": direct_image
                        })
                    elif hasattr(direct_image, 'convert'):
                        image_data = resize_image_if_needed(direct_image)
                        
                        with NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                            temp_path = temp_file.name
                            image_data.save(temp_path, format='JPEG')
                        
                        image_data = process_image_with_vision_module(image_data)
                        print(f"Processed PIL image: {type(image_data)}")
                                
                        user_content.append({
                            "type": "image",
                            "image": temp_path
                        })
                except Exception as e:
                    print(f"Error processing image: {e}")
                    traceback_str = traceback.format_exc()
                    print(f"Traceback: {traceback_str}")
            
            if user_message:
                user_content.append({
                    "type": "text",
                    "text": user_message
                })
            
            if not user_content:
                user_content.append({
                    "type": "text",
                    "text": "Hello"
                })
            
            conversation.append({
                "role": "user",
                "content": user_content
            })
            
            print(f"Conversation structure: {conversation}")
            
            try:
                if len(conversation) < 2:
                    if len(conversation) == 1 and conversation[0]["role"] != "system":
                        conversation.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
                    elif len(conversation) == 0:
                        conversation = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
                        ]
                
                system_msg = next((msg for msg in conversation if msg["role"] == "system"), {"role": "system", "content": SYSTEM_PROMPT})
                user_msg = conversation[-1]

                text = self.processor.apply_chat_template(
                    [system_msg, user_msg], 
                    add_generation_prompt=True, 
                    tokenize=False
                )
                image_data = None
                
                if "content" in user_msg and isinstance(user_msg["content"], list):
                    for content in user_msg["content"]:
                        if content.get("type") == "image" and "image" in content:
                            try:
                                image_path = content["image"]
                                if os.path.exists(image_path):
                                    # First load with PIL and resize
                                    pil_image = Image.open(image_path)
                                    # Process using our helper function with correct method names
                                    image_data = process_image_with_vision_module(pil_image)
                                    print(f"Processed image with helper function: {type(image_data)}")
                            except Exception as e:
                                print(f"Error loading image: {e}")
                                image_data = None
                
                has_image = image_data is not None
                
                if has_image:
                    print(f"Creating multimedia inputs: image={has_image}")
                    try:
                        inputs = self.processor(
                            text=text,
                            images=image_data,
                            return_tensors="pt"
                        )
                        
                        model_dtype = next(self.model.parameters()).dtype
                        print(f"Model is using dtype: {model_dtype}")
                        
                        inputs_converted = {}
                        for k, v in inputs.items():
                            if isinstance(v, torch.Tensor):
                                inputs_converted[k] = v.to(device=self.model.device)
                            else:
                                inputs_converted[k] = v
                        
                        inputs = inputs_converted
                        print("Successfully created inputs object with image")
                    except Exception as e:
                        print(f"Error preparing model inputs: {e}")

                        traceback_str = traceback.format_exc()
                        print(f"Traceback: {traceback_str}")
                        print("Using emergency fallback inputs")

                        inputs = self.processor(
                            text=text,
                            return_tensors="pt"
                        ).to(self.model.device)
                else:
                    print("Creating text-only inputs")
                    inputs = self.processor(
                        text=text,
                        return_tensors="pt"
                    ).to(self.model.device)
                
                try:
                    generation_output = self.model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=self.temperature,
                        max_new_tokens=self.max_new_tokens,
                    )
                except Exception as e:
                    print(f"Error in generation: {e}")
                    
                    traceback_str = traceback.format_exc()
                    print(f"Traceback: {traceback_str}")
                    
                    try:
                        print("Creating text-only backup inputs")
                        backup_inputs = self.processor(
                            text=text,
                            return_tensors="pt"
                        ).to(self.model.device)
                        
                        generation_output = self.model.generate(
                            **backup_inputs,
                            do_sample=True,
                            temperature=self.temperature,
                            max_new_tokens=self.max_new_tokens,
                        )
                    except Exception as e2:
                        print(f"Error in backup generation: {e2}")
                        return "I'm having trouble processing your request. Please try again with text only."
                
                if isinstance(generation_output, tuple) and len(generation_output) >= 2:
                    print("Processing generation output tuple")

                    text_ids = generation_output[0]
                    audio = generation_output[1]
                elif hasattr(generation_output, "sequences"):
                    print("Processing generation output with sequences attribute")

                    text_ids = generation_output.sequences
                    audio = None
                else:
                    print(f"Unexpected generation output type: {type(generation_output)}")

                    text_ids = generation_output
                    audio = None

                generated_text = None
                
                try:
                    print("Trying processor.decode")
                    generated_text = self.processor.decode(text_ids[0], skip_special_tokens=True)
                except Exception as e1:
                    print(f"Error with processor.decode: {e1}")
                    try:
                        print("Trying tokenizer.decode")
                        generated_text = self.tokenizer.decode(text_ids[0], skip_special_tokens=True)
                    except Exception as e2:
                        print(f"Error with tokenizer.decode: {e2}")
                        try:
                            print("Trying processor.batch_decode")
                            generated_text = self.processor.batch_decode(text_ids, skip_special_tokens=True)[0]
                        except Exception as e3:
                            print(f"Error with processor.batch_decode: {e3}")
                            generated_text = "I'm having trouble generating a response right now."
                
                print(f"Generated text (sample): {generated_text[:100] if generated_text else 'None'}...")
                
                assistant_message = self.extract_assistant_message(generated_text, text)
                print(f"Extracted assistant message (sample): {assistant_message[:100] if assistant_message else 'None'}...")
                
                output_audio_path = None
                if audio is not None:
                    print("Generating audio from model output")

                    with NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        output_audio_path = f.name
                        
                    try:
                        if hasattr(audio, "audio_array") and audio.audio_array is not None:
                            audio_array = audio.audio_array.cpu().numpy()
                            sample_rate = audio.sampling_rate
                        else:
                            audio_array = audio.float().cpu().numpy()
                            sample_rate = 24000
                        
                        sf.write(output_audio_path, audio_array, sample_rate)
                        print(f"Generated audio saved to: {output_audio_path}")
                    except Exception as audio_e:
                        print(f"Error generating audio: {audio_e}")
                        output_audio_path = None
                
                assistant_response = {"role": "assistant", "content": assistant_message}
                new_history = conversation + [assistant_response]
                display_messages = self.format_messages_for_gradio([conversation[-1], assistant_response])
                
                return new_history, display_messages, output_audio_path
                
            except Exception as gen_e:
                print(f"Error in generation: {gen_e}")
                traceback_str = traceback.format_exc()
                print(f"Traceback: {traceback_str}")
                
                fallback_message = "I'm having trouble processing your request. "
                
                if len(images) > 0 and len(audios) > 0:
                    fallback_message += "Processing both images and audio is challenging. "
                elif len(images) > 0:
                    fallback_message += "The image processing capability is still being refined. "
                elif len(audios) > 0:
                    fallback_message += "The audio processing capability is still being refined. "
                
                fallback_message += "The Qwen 2.5 Omni model is very new and its multi-modal capabilities in transformers are still evolving. Text-only questions work best for now."
                
                if str(gen_e):
                    fallback_message += f"\n\nError details: {str(gen_e)}"
                
                assistant_response = {"role": "assistant", "content": fallback_message}
                
                new_history = conversation + [assistant_response]
                display_messages = self.format_messages_for_gradio([conversation[-1], assistant_response])
                
                return new_history, display_messages, None
                
        except Exception as e:

            print(f"Error in generate_response: {e}")
            traceback_str = traceback.format_exc()
            print(f"Traceback: {traceback_str}")
            
            error_message = f"Error generating response: {e}"
            assistant_response = {"role": "assistant", "content": error_message}
            
            if 'conversation' in locals() and conversation:
                new_history = conversation + [assistant_response]
            else:
                user_content = [{"type": "text", "text": user_message or "Hello"}]
                user_msg = {"role": "user", "content": user_content}
                new_history = [{"role": "system", "content": SYSTEM_PROMPT}, user_msg, assistant_response]
            
            display_messages = self.format_messages_for_gradio([new_history[-2], assistant_response])
            
            return new_history, display_messages, None

    def process_mm_info(self, conversation, use_audio_in_video=False):
        """Process multimedia info in the messages."""
        try:
            if QWEN_UTILS_AVAILABLE:
                import qwen_omni_utils
                print("Using qwen_omni_utils.process_mm_info")
                try:
                    print(f"Conversation to process: {conversation}")
                    
                    direct_images = []
                    for msg in conversation:
                        if msg["role"] == "user" and isinstance(msg.get("content"), list):
                            for content in msg["content"]:
                                if content.get("type") == "image" and "image" in content:
                                    print(f"Found direct image in message: {content.get('image')[:30]}...")

                                    try:
                                        if isinstance(content["image"], str) and os.path.exists(content["image"]):
                                            img = Image.open(content["image"])
                                            img = resize_image_if_needed(img)
                                            direct_images.append(img)

                                            print(f"Successfully loaded image from path: {content['image']}")
                                    except Exception as img_e:
                                        print(f"Error loading direct image: {img_e}")
                    
                    result = qwen_omni_utils.process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
                    
                    if isinstance(result, tuple) and len(result) == 3:
                        audios, images, videos = result
                        print(f"Got processed media: audios={len(audios)}, images={len(images)}, videos={len(videos)}")
                    else:
                        print(f"Unexpected return value from qwen_omni_utils.process_mm_info: {type(result)}")
                        audios, images, videos = [], [], []
                    if not images and direct_images:
                        print(f"Using {len(direct_images)} direct images instead of processed ones")
                        images = direct_images
                except Exception as e:
                    print(f"Error in qwen_omni_utils.process_mm_info: {e}")

                    traceback_str = traceback.format_exc()
                    print(f"Traceback: {traceback_str}")
                    if direct_images:
                        images = direct_images
                    else:
                        images = []
                    audios, videos = [], []
                
                text = self.processor.apply_chat_template(
                    conversation, 
                    add_generation_prompt=True, 
                    tokenize=False
                )
                
                has_multimedia = (len(images) > 0 or len(audios) > 0 or len(videos) > 0)
                if has_multimedia:
                    print(f"Creating multimedia inputs: images={len(images)}, audios={len(audios)}, videos={len(videos)}")
                    
                    if images:
                        print(f"First image type: {type(images[0])}")
                        if isinstance(images[0], str) and os.path.exists(images[0]):
                            print(f"Loading image from path: {images[0]}")
                            images[0] = Image.open(images[0])
                            images[0] = resize_image_if_needed(images[0])
                    
                    inputs = self.processor(
                        text=text,
                        images=images[0] if images else None,
                        audios=audios[0] if audios else None,
                        videos=videos[0][0] if videos and videos[0] else None,
                        return_tensors="pt"
                    )
                else:
                    print("Creating text-only inputs")
                    inputs = self.processor(
                        text=text,
                        return_tensors="pt"
                    )
                
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                return text, inputs
            else:
                print("qwen_omni_utils not available, using simple text processing")
                text = self.processor.apply_chat_template(
                    conversation, 
                    add_generation_prompt=True, 
                    tokenize=False
                )
                
                inputs = self.processor(
                    text=text,
                    return_tensors="pt"
                )
                
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                return text, inputs
                
        except Exception as e:
            print(f"Error in process_mm_info: {e}")
            traceback_str = traceback.format_exc()
            print(f"Traceback: {traceback_str}")
            
            text = ""
            if conversation and len(conversation) > 0:
                try:
                    text = self.processor.apply_chat_template(
                        conversation, 
                        add_generation_prompt=True, 
                        tokenize=False
                    )
                except Exception as template_e:
                    print(f"Error applying chat template: {template_e}")
                    text = "Hello, how can I help you?"
            else:
                text = "Hello, how can I help you?"
            
            try:
                inputs = self.processor(
                    text=text,
                    return_tensors="pt"
                )
                
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            except Exception as proc_e:
                print(f"Error in fallback processor: {proc_e}")
                inputs = {
                    "input_ids": torch.tensor([[1]]).to(self.model.device),
                    "attention_mask": torch.tensor([[1]]).to(self.model.device)
                }
            
            return text, inputs

    def extract_assistant_message(self, generated_text, prompt_text):
        """Extract the assistant's message from the generated text."""
        try:
            if isinstance(prompt_text, list):
                prompt_text = str(prompt_text)
            if not isinstance(generated_text, str):
                generated_text = str(generated_text)
            
            print(f"Extracting from generated text: {generated_text[:50]}...")
            
            if "assistant" in generated_text:
                print("Found assistant marker")

                parts = generated_text.split("assistant")
                if len(parts) > 1:
                    assistant_part = parts[-1].strip()
                    if "assistant" in assistant_part:
                        assistant_text = assistant_part.split("assistant")[0].strip()
                    else:
                        assistant_text = assistant_part.strip()
            elif "ASSISTANT:" in generated_text:
                print("Found ASSISTANT: marker")
                assistant_text = generated_text.split("ASSISTANT:")[-1].strip()
            elif prompt_text and prompt_text in generated_text:
                print("Found prompt text in response")
                assistant_text = generated_text[generated_text.find(prompt_text) + len(prompt_text):].strip()
            else:
                print("No markers found, using as-is")
                assistant_text = generated_text.strip()

            assistant_text = self.clean_text(assistant_text)
            print(f"Extracted assistant message: {assistant_text[:50]}...")
            return assistant_text
        except Exception as e:
            print(f"Error in extract_assistant_message: {e}")
            traceback_str = traceback.format_exc()
            print(f"Traceback: {traceback_str}")

            return "I'm having trouble generating a response."

    def clean_text(self, text):
        """Clean the generated text to remove unwanted content."""
        if text is None:
            return ""

        prefixes_to_remove = [
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            "system", "user", "USER:", "SYSTEM:", ":"
        ]
        
        cleaned_text = text
        for prefix in prefixes_to_remove:
            if cleaned_text.lower().startswith(prefix.lower()):
                cleaned_text = cleaned_text[len(prefix):].lstrip()
                
        if cleaned_text.lower().startswith("assistant:"):
            cleaned_text = cleaned_text[len("assistant:"):].lstrip()
        
        paragraphs = cleaned_text.split('\n\n')
        if len(paragraphs) > 1 and len(paragraphs[0]) < 30:
            if any(marker in paragraphs[0].lower() for marker in ["system", "user", "assistant"]):
                paragraphs = paragraphs[1:]
                cleaned_text = '\n\n'.join(paragraphs)
        
        common_artifacts = [
            "system", "user", "assistant", "SYSTEM:", "USER:", "ASSISTANT:", 
            "You are Qwen,", "a virtual human"
        ]
        
        for artifact in common_artifacts:
            if cleaned_text.startswith(artifact):
                cleaned_text = cleaned_text[len(artifact):].lstrip()
        
        return cleaned_text

    def format_messages_for_gradio(self, messages):
        """Format messages for display in Gradio UI."""
        try:
            if messages is None:
                print("Warning: messages is None in format_messages_for_gradio")
                return []
                
            formatted_messages = []
            
            for message in messages:
                if not isinstance(message, dict):
                    print(f"Warning: skipping non-dict message: {type(message)}")
                    continue
                    
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "system":
                    continue
                
                if role == "user":
                    if isinstance(content, list):
                        text_items = [item for item in content if isinstance(item, dict) and item.get("type") == "text"]
                        if text_items:
                            display_text = text_items[0].get("text", "")
                        else:
                            display_text = "(Image or Audio Input)"
                    else:
                        display_text = str(content)
                    
                    formatted_messages.append({"role": "user", "content": display_text})
                
                elif role == "assistant":
                    if isinstance(content, list):
                        text_content = " ".join([
                            item.get("text", "") 
                            for item in content 
                            if isinstance(item, dict) and item.get("type") == "text"
                        ])
                        formatted_messages.append({"role": "assistant", "content": text_content})
                    else:
                        formatted_messages.append({"role": "assistant", "content": str(content)})
            return formatted_messages
            
        except Exception as e:
            print(f"Error in format_messages_for_gradio: {e}")
            traceback_str = traceback.format_exc()
            print(f"Traceback: {traceback_str}")
            return []

def create_demo():
    """Create the Gradio demo."""
    ui = MultiModalUI()
    
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.Markdown(
            """
            <div align="center">
            <h1>Multimodal Web UI</h1>
            <p>Chat with Qwen 2.5 Omni model that can process text, images, audio, and video.</p>
            </div>
            """
        )
        
        chatbot_state = gr.State([])
        
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Accordion("Upload Media", open=False):
                    with gr.Tab("Images"):
                        image_input = gr.Image(
                            type="filepath",
                            label="Upload Image",
                            sources=["upload", "clipboard"],
                            elem_id="image_upload",
                            elem_classes="image-upload-component"
                        )
                    with gr.Tab("Audio"):
                        gr.Markdown("## Audio Input (Coming Soon)")
                        audio_input = gr.State(None)
                    with gr.Tab("Video"):
                        gr.Markdown("## Video Input (Coming Soon)")
                        video_input = gr.State(None)

                chat_interface = gr.Chatbot(
                    height=500,
                    show_copy_button=True,
                    avatar_images=None,
                    value=[],
                    type="messages"
                )
                
                with gr.Row():
                    with gr.Column(scale=8):
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="Type your message here...",
                            container=False
                        )
                    with gr.Column(scale=1):
                        submit_btn = gr.Button("Send")
                    with gr.Column(scale=1):
                        clear_btn = gr.Button("Clear")

                audio_output = gr.Audio(type="filepath", label="Generated Speech", visible=True)
        
        def on_submit(message, image_input, history):
            """Process user input and generate a response."""
            print(f"on_submit called: message={message}, image_path={image_input}, history={len(history) if history else 0}")
            
            try:

                if history is None:
                    history = []
                    
                if len(history) > 50:
                    print(f"Warning: Very large history detected ({len(history)} messages), resetting to empty")
                    history = []
                
                if image_input is not None:
                    print("Multimedia input detected, creating fresh conversation")
                    history = []

                direct_image = None

                image_path = image_input
                if image_path:
                    try:
                        direct_image = Image.open(image_path)
                        print(f"Successfully loaded image: {image_path}")
                    except Exception as e:
                        print(f"Error loading image: {e}")
                
                print(f"Calling generate_response with message={message}, direct_image={direct_image is not None}, history_len={len(history)}")
                new_history, display_messages, audio_path = ui.generate_response(message, direct_image, None, history)
            
                return "", image_input, new_history, display_messages, audio_path
            except Exception as e:
                traceback_str = traceback.format_exc()
                print(f"Error in on_submit: {e}")
                print(f"Traceback: {traceback_str}")
                return "", image_input, history, [["Error occurred", str(e)]], None
        
        def clear_conversation():
            """Clear the conversation history completely."""
            print("Clearing conversation history")
            ui.reset_state()
            return None, [], [], None
        
        submit_btn.click(
            on_submit,
            inputs=[msg, image_input, chatbot_state],
            outputs=[msg, image_input, chatbot_state, chat_interface, audio_output],
        )
        
        msg.submit(
            on_submit,
            inputs=[msg, image_input, chatbot_state],
            outputs=[msg, image_input, chatbot_state, chat_interface, audio_output],
        )
        
        clear_btn.click(
            clear_conversation, 
            outputs=[image_input, chat_interface, chatbot_state, audio_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", share=True)
