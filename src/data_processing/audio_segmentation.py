import librosa
import soundfile as sf
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Segment processor for creating shorter audio segments
class AudioSegmenter:
    """Split longer audio files into segments"""
    
    def __init__(self, segment_length=30, overlap=5):
        """
        Initialize audio segmenter
        
        Args:
            segment_length: Length of each segment in seconds
            overlap: Overlap between segments in seconds
        """
        self.segment_length = segment_length
        self.overlap = overlap
    
    def segment_audio(self, audio_path, output_dir):
        """
        Segment audio file into smaller chunks
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save segments
            
        Returns:
            DataFrame with segment metadata
        """
        # Create episode-specific folder
        audio_name = Path(audio_path).stem
        episode_dir = os.path.join(output_dir, f"segment_{audio_name}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculate segment parameters
        segment_samples = self.segment_length * sr
        overlap_samples = self.overlap * sr
        step_size = segment_samples - overlap_samples
        
        # Create segments
        segments = []
        for i, start_sample in enumerate(range(0, len(y), step_size)):
            # Calculate end position
            end_sample = min(start_sample + segment_samples, len(y))
            
            # Skip if segment is too short
            if end_sample - start_sample < segment_samples * 0.5:
                break
            
            # Extract segment
            segment = y[start_sample:end_sample]
            
            # Create segment name
            segment_path = os.path.join(episode_dir, f"{audio_name}_seg{i:03d}.wav")
            
            # Save segment
            sf.write(segment_path, segment, sr)
            
            # Store metadata
            segments.append({
                'original_file': audio_path,
                'episode_folder': episode_dir,
                'segment_id': f"{audio_name}_seg{i:03d}",
                'segment_path': segment_path,
                'start_time': start_sample / sr,
                'end_time': end_sample / sr,
                'duration': (end_sample - start_sample) / sr
            })
        
        # Create episode-specific metadata
        segments_df = pd.DataFrame(segments)
        if not segments_df.empty:
            segments_df.to_csv(os.path.join(episode_dir, f"{audio_name}_segments.csv"), index=False)
        
        return segments_df
    
    def segment_dataset(self, metadata_df, output_dir):
        """
        Segment all audio files in a dataset
        
        Args:
            metadata_df: DataFrame with audio file metadata
            output_dir: Directory to save segments
            
        Returns:
            DataFrame with segment metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_segments = []
        
        for idx, row in tqdm(metadata_df.iterrows(), desc="Segmenting audio", total=len(metadata_df)):
            try:
                segments = self.segment_audio(row['processed_path'], output_dir)
                all_segments.append(segments)
            except Exception as e:
                print(f"Error segmenting {row['processed_path']}: {str(e)}")
        
        if all_segments:
            segments_df = pd.concat(all_segments, ignore_index=True)
            segments_df.to_csv(os.path.join(output_dir, 'segments_metadata.csv'), index=False)
            return segments_df
        
        return pd.DataFrame()

    def segment_files_directly(self, audio_dir, output_dir, file_pattern='*.wav'):
        """
        Segment all audio files in a directory directly without metadata
        
        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory to save segments
            file_pattern: Pattern to match audio files (default: *.wav)
            
        Returns:
            DataFrame with segment metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all audio files
        audio_paths = list(Path(audio_dir).glob(file_pattern))
        
        if not audio_paths:
            print(f"No audio files matching '{file_pattern}' found in {audio_dir}")
            return pd.DataFrame()
        
        print(f"Found {len(audio_paths)} audio files to segment")
        
        all_segments = []
        for audio_path in tqdm(audio_paths, desc="Segmenting audio"):
            try:
                segments = self.segment_audio(str(audio_path), output_dir)
                all_segments.append(segments)
            except Exception as e:
                print(f"Error segmenting {audio_path}: {str(e)}")
        
        if all_segments:
            segments_df = pd.concat(all_segments, ignore_index=True)
            segments_df.to_csv(os.path.join(output_dir, 'segments_metadata.csv'), index=False)
            return segments_df
        
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Define directories
    audio_dir = "../../data/raw"  # Directory containing original audio files
    output_dir = "../../data/processed/segments"  # Directory to save segments
    
    # Create segmenter
    segmenter = AudioSegmenter(segment_length=30, overlap=5)
    
    # Look for audio files with various extensions
    all_segments = []
    for file_pattern in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
        print(f"\nLooking for {file_pattern} files...")
        segments = segmenter.segment_files_directly(
            audio_dir=audio_dir,
            output_dir=output_dir,
            file_pattern=file_pattern
        )
        
        if not segments.empty:
            print(f"Created {len(segments)} segments from {file_pattern} files")
            all_segments.append(segments)
    
    # Create a combined metadata file with all segments
    if all_segments:
        all_segments_df = pd.concat(all_segments, ignore_index=True)
        all_segments_df.to_csv(os.path.join(output_dir, 'all_segments_metadata.csv'), index=False)
        print(f"\nTotal segments created: {len(all_segments_df)}")
        print(f"Episode folders created: {all_segments_df['episode_folder'].nunique()}")