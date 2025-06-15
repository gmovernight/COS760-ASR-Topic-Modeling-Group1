import os
import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Wav2VecTranscriber:
    def __init__(self):
        """Initialize the wav2vec transcriber with facebook/mms-1b-all model"""
        self.processor = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the wav2vec model using default cache location"""
        model_name = "facebook/mms-1b-all"
        
        try:
            logger.info(f"Loading model: {model_name}")
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise RuntimeError(f"Could not load wav2vec model: {e}")
    
    def preprocess_audio(self, audio_path):
        """
        Load and preprocess audio file for wav2vec processing
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio tensor
        """
        try:
            # Load audio
            audio, sample_rate = sf.read(audio_path)
            
            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                except ImportError:
                    logger.error("librosa not installed. Install with: pip install librosa")
                    raise ValueError("Audio must be sampled at 16kHz or install librosa for resampling")
            
            # Preprocess for model
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            return inputs
            
        except Exception as e:
            logger.error(f"Failed to preprocess audio {audio_path}: {e}")
            raise
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription text
        """
        try:
            # Preprocess audio
            inputs = self.preprocess_audio(audio_path)
            
            # Perform inference
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
            
            # Decode to text
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            return transcription
            
        except Exception as e:
            logger.error(f"Failed to transcribe {audio_path}: {e}")
            return ""
    
    def transcribe_episode_segments(self, episode_folder, output_dir):
        """
        Transcribe all segments from an episode and combine into one transcript
        
        Args:
            episode_folder: Path to folder containing episode segments
            output_dir: Directory to save transcripts
            
        Returns:
            Path to saved transcript file
        """
        episode_name = Path(episode_folder).name.replace("segment_", "")
        
        # Create output directory for this episode
        episode_output_dir = os.path.join(output_dir, episode_name)
        os.makedirs(episode_output_dir, exist_ok=True)
        
        # Find all audio segments in the episode folder
        segment_files = []
        for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
            segment_files.extend(Path(episode_folder).glob(ext))
        
        # Sort segments by name to maintain order
        segment_files.sort()
        
        if not segment_files:
            logger.warning(f"No audio segments found in {episode_folder}")
            return None
        
        logger.info(f"Transcribing {len(segment_files)} segments for episode: {episode_name}")
        
        # Transcribe each segment
        full_transcription = []
        for segment_file in tqdm(segment_files, desc=f"Transcribing {episode_name}"):
            try:
                transcription = self.transcribe_audio(str(segment_file))
                if transcription.strip():  # Only add non-empty transcriptions
                    full_transcription.append(transcription.strip())
            except Exception as e:
                logger.error(f"Failed to transcribe segment {segment_file}: {e}")
                continue
        
        # Combine all transcriptions
        combined_transcription = " ".join(full_transcription)
        
        # Save to file
        output_file = os.path.join(episode_output_dir, f"{episode_name}_wav2vec_transcript.txt")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(combined_transcription)
            logger.info(f"Saved transcript to: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Failed to save transcript: {e}")
            return None
    
    def transcribe_all_episodes(self, segments_dir, output_dir):
        """
        Transcribe all episodes in the segments directory
        
        Args:
            segments_dir: Directory containing episode segment folders
            output_dir: Directory to save transcripts
            
        Returns:
            List of transcript file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all episode folders (folders starting with "segment_")
        episode_folders = [f for f in Path(segments_dir).iterdir() 
                          if f.is_dir() and f.name.startswith("segment_")]
        
        if not episode_folders:
            logger.error(f"No episode folders found in {segments_dir}")
            return []
        
        logger.info(f"Found {len(episode_folders)} episodes to transcribe")
        
        transcript_files = []
        for episode_folder in episode_folders:
            try:
                transcript_file = self.transcribe_episode_segments(str(episode_folder), output_dir)
                if transcript_file:
                    transcript_files.append(transcript_file)
            except Exception as e:
                logger.error(f"Failed to process episode {episode_folder}: {e}")
                continue
        
        logger.info(f"Successfully transcribed {len(transcript_files)} episodes")
        return transcript_files

# Example usage
if __name__ == "__main__":
    # Define directories
    segments_dir = "../../data/processed/segments"  # Directory containing episode segments
    output_dir = "../../data/transcripts/wav2vec"   # Directory to save transcripts
    
    # Create transcriber
    transcriber = Wav2VecTranscriber()
    
    # Transcribe all episodes
    transcript_files = transcriber.transcribe_all_episodes(segments_dir, output_dir)
    
    print(f"\nTranscription complete!")
    print(f"Transcribed {len(transcript_files)} episodes")
    print(f"Transcripts saved to: {output_dir}")