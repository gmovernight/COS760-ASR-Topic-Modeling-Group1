import whisper
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

class WhisperTranscriber:
    """Simple Whisper transcriber for audio segments"""
    
    def __init__(self, model_size='large', device=None, language=None):
        """
        Initialize Whisper transcriber
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            language: Target language code (e.g., 'en', 'es') or None for auto-detect
        """
        self.model_size = model_size
        self.language = language
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Whisper model '{model_size}' on device '{self.device}'...")
        self.model = whisper.load_model(model_size, device=self.device)
        print("Model loaded successfully!")
    
    def transcribe_segments_by_folder(self, segments_df, base_output_dir="../../data/transcripts/whisper"):
        """
        Transcribe segments and save combined text file per folder
        
        Args:
            segments_df: DataFrame with segment metadata containing 'folder_name' and 'segment_path'
            base_output_dir: Base directory to save transcriptions
            
        Returns:
            Dictionary with transcription results by folder
        """
        results = {}
        
        # Group segments by folder
        folder_groups = segments_df.groupby('folder_name')
        
        for folder_name, group in folder_groups:
            print(f"\nProcessing folder: {folder_name}")
            
            # Create output directory for this folder
            folder_output_dir = Path(base_output_dir) / folder_name
            folder_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Output file path
            txt_file_path = folder_output_dir / f"{folder_name}.txt"
            
            # Sort segments by start time to maintain chronological order
            sorted_group = group.sort_values('start_time')
            
            # Transcribe all segments in this folder
            folder_transcriptions = []
            
            for idx, row in tqdm(sorted_group.iterrows(), 
                               desc=f"Transcribing {folder_name}", 
                               total=len(sorted_group)):
                
                try:
                    # Transcribe segment
                    result = self.model.transcribe(
                        row['segment_path'],
                        language=self.language,
                        fp16=self.device == 'cuda'
                    )
                    
                    transcription_text = result['text'].strip()
                    
                    if transcription_text:  # Only add if there's actual text
                        folder_transcriptions.append({
                            'segment_id': row['segment_id'],
                            'start_time': row['start_time'],
                            'end_time': row['end_time'],
                            'text': transcription_text
                        })
                        
                except Exception as e:
                    print(f"Error transcribing {row['segment_path']}: {str(e)}")
                    continue
            
            # Save combined transcription to text file
            if folder_transcriptions:
                with open(txt_file_path, 'w', encoding='utf-8') as f:
                    clean_text = " ".join([trans['text'] for trans in folder_transcriptions])
                    f.write(clean_text)
                
                results[folder_name] = {
                    'file_path': str(txt_file_path),
                    'segment_count': len(folder_transcriptions),
                    'total_words': len(clean_text.split()),
                    'transcriptions': folder_transcriptions
                }
                
                print(f"✅ Saved transcription: {txt_file_path}")
                print(f"   - {len(folder_transcriptions)} segments transcribed")
                print(f"   - {len(clean_text.split())} total words")
            
            else:
                print(f"❌ No valid transcriptions found for folder: {folder_name}")
                results[folder_name] = {
                    'file_path': None,
                    'segment_count': 0,
                    'total_words': 0,
                    'transcriptions': []
                }
        
        return results
    
    def transcribe_folder_from_csv(self, csv_path, base_output_dir="../../data/transcripts/whisper"):
        """
        Convenience method to transcribe from a segments CSV file
        
        Args:
            csv_path: Path to segments metadata CSV file
            base_output_dir: Base directory to save transcriptions
            
        Returns:
            Dictionary with transcription results by folder
        """
        print(f"Loading segments from: {csv_path}")
        segments_df = pd.read_csv(csv_path)
        
        required_columns = ['folder_name', 'segment_path', 'segment_id', 'start_time', 'end_time']
        missing_columns = [col for col in required_columns if col not in segments_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        print(f"Found {len(segments_df)} segments across {segments_df['folder_name'].nunique()} folders")
        
        return self.transcribe_segments_by_folder(segments_df, base_output_dir)


def transcribe_audio_segments(segments_csv_path, output_dir="../../data/transcripts/whisper", 
                              model_size='large', language=None):
    """
    Simple function to transcribe audio segments and save to text files
    
    Args:
        segments_csv_path: Path to CSV file with segment metadata
        output_dir: Output directory (default: "../../data/transcripts/whisper")
        model_size: Whisper model size (default: 'large')
        language: Language code or None for auto-detect
        
    Returns:
        Dictionary with results
    """
    # Initialize transcriber
    transcriber = WhisperTranscriber(
        model_size=model_size,
        language=language
    )
    
    # Transcribe segments
    results = transcriber.transcribe_folder_from_csv(
        csv_path=segments_csv_path,
        base_output_dir=output_dir
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRANSCRIPTION SUMMARY")
    print(f"{'='*60}")
    
    total_segments = sum(r['segment_count'] for r in results.values())
    total_words = sum(r['total_words'] for r in results.values())
    successful_folders = len([r for r in results.values() if r['segment_count'] > 0])
    
    print(f"Total folders processed: {len(results)}")
    print(f"Successful folders: {successful_folders}")
    print(f"Total segments transcribed: {total_segments}")
    print(f"Total words: {total_words}")
    print(f"Output directory: {output_dir}")
    
    print(f"\nFiles created:")
    for folder_name, result in results.items():
        if result['file_path']:
            print(f"  - {result['file_path']}")
    
    return results


if __name__ == "__main__":
    results = transcribe_audio_segments(
        segments_csv_path="../../data/processed/segments/segments_metadata.csv",
        output_dir="../../data/transcripts/whisper",
        model_size='large',
        language='Swahili'
    )
