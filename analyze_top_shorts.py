"""
Analyze Top YouTube Shorts
This script fetches the top 100 most-viewed shorts from a YouTube channel,
downloads each video, analyzes it with Gemini AI, and creates a dataset
of titles and their effectiveness.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import time

# Third-party imports
try:
    from googleapiclient.discovery import build # type: ignore
    from googleapiclient.errors import HttpError # type: ignore
    import yt_dlp # type: ignore
    import google.generativeai as genai # type: ignore
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install google-api-python-client yt-dlp google-generativeai")
    sys.exit(1)

# Local imports
from utils.config import get_gemini_api_key


class YouTubeShortAnalyzer:
    """Handles fetching, downloading, and analyzing YouTube shorts."""
    
    def __init__(self, channel_url, output_file="shorts_analysis.json", max_shorts=100):
        """
        Initialize the analyzer.
        
        Args:
            channel_url: URL of the YouTube channel shorts page
            output_file: Path to save JSON results
            max_shorts: Maximum number of shorts to analyze (default 100)
        """
        self.channel_url = channel_url
        self.output_file = output_file
        self.max_shorts = max_shorts
        self.temp_dir = Path("temp_short_download")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Load API keys from config
        self.youtube_api_key = self._load_youtube_api_key()
        self.gemini_api_key = get_gemini_api_key()
        
        # Initialize APIs
        self.youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
        genai.configure(api_key=self.gemini_api_key)
        
        # Safety settings to prevent unnecessary blocking
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        
        self.gemini_model = genai.GenerativeModel(
            'models/gemini-2.5-flash',
            safety_settings=self.safety_settings
        )
        
        # Extract channel handle from URL
        self.channel_handle = self._extract_channel_handle(channel_url)
        
    def _load_youtube_api_key(self):
        """Load YouTube API key from config.json"""
        config_path = Path(__file__).parent / 'utils' / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError("config.json not found. Please add YOUTUBE_API_KEY to config.")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        api_key = config.get("YOUTUBE_API_KEY")
        if not api_key:
            raise ValueError("YOUTUBE_API_KEY not found in config.json")
        
        return api_key
    
    def _extract_channel_handle(self, url):
        """Extract channel handle from URL."""
        # URL format: https://www.youtube.com/@PeepingOtter/shorts
        parts = url.rstrip('/').split('/')
        for part in parts:
            if part.startswith('@'):
                return part
        raise ValueError(f"Could not extract channel handle from URL: {url}")
    
    def fetch_top_shorts(self):
        """
        Fetch top shorts from the channel sorted by view count.
        
        Returns:
            List of dicts with video information
        """
        print(f"Fetching shorts from channel: {self.channel_handle}")
        
        try:
            # First, get the channel ID from the handle
            channel_response = self.youtube.channels().list(
                part='id,contentDetails',
                forHandle=self.channel_handle.lstrip('@'),
                maxResults=1
            ).execute()
            
            if not channel_response.get('items'):
                raise ValueError(f"Channel not found: {self.channel_handle}")
            
            channel_id = channel_response['items'][0]['id']
            print(f"Found channel ID: {channel_id}")
            
            # Fetch ALL videos from the channel first
            # We need to get all videos, filter for shorts, then sort by views
            print("Fetching all videos from channel...")
            all_videos = []
            next_page_token = None
            page_count = 0
            
            while True:
                page_count += 1
                search_response = self.youtube.search().list(
                    part='id',
                    channelId=channel_id,
                    type='video',
                    maxResults=50,
                    pageToken=next_page_token,
                    order='date'  # Get by date to ensure we get all videos
                ).execute()
                
                video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
                
                if not video_ids:
                    break
                
                all_videos.extend(video_ids)
                print(f"  Page {page_count}: Found {len(video_ids)} videos (total: {len(all_videos)})")
                
                next_page_token = search_response.get('nextPageToken')
                if not next_page_token:
                    break
            
            print(f"Total videos found: {len(all_videos)}")
            
            # Now get details for all videos and filter for shorts
            print("Filtering for shorts and getting view counts...")
            shorts = []
            
            # Process in batches of 50 (API limit)
            for i in range(0, len(all_videos), 50):
                batch = all_videos[i:i+50]
                
                videos_response = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    id=','.join(batch)
                ).execute()
                
                for video in videos_response.get('items', []):
                    duration = video['contentDetails']['duration']
                    
                    # Only include if it's a short (under 60 seconds)
                    if self._is_short_duration(duration):
                        shorts.append({
                            'video_id': video['id'],
                            'title': video['snippet']['title'],
                            'views': int(video['statistics'].get('viewCount', 0)),
                            'published_date': video['snippet']['publishedAt'].split('T')[0],
                            'duration': duration,
                            'url': f"https://www.youtube.com/shorts/{video['id']}"
                        })
                
                print(f"  Processed {min(i+50, len(all_videos))}/{len(all_videos)} videos, found {len(shorts)} shorts so far")
            
            print(f"\nTotal shorts found: {len(shorts)}")
            
            # Sort by views (descending) and take top N
            shorts.sort(key=lambda x: x['views'], reverse=True)
            top_shorts = shorts[:self.max_shorts]
            
            print(f"Selected top {len(top_shorts)} shorts by view count")
            if top_shorts:
                print(f"  Highest views: {top_shorts[0]['views']:,}")
                print(f"  Lowest views: {top_shorts[-1]['views']:,}")
            
            return top_shorts
            
        except HttpError as e:
            print(f"YouTube API error: {e}")
            raise
    
    def _is_short_duration(self, duration_str):
        """
        Check if video duration indicates it's a short (under 60 seconds).
        Duration format: PT#M#S or PT#S
        """
        # Remove PT prefix
        duration = duration_str.replace('PT', '')
        
        # If it has hours, it's not a short
        if 'H' in duration:
            return False
        
        # Parse minutes and seconds
        minutes = 0
        seconds = 0
        
        if 'M' in duration:
            parts = duration.split('M')
            minutes = int(parts[0])
            if len(parts) > 1 and 'S' in parts[1]:
                seconds = int(parts[1].replace('S', ''))
        elif 'S' in duration:
            seconds = int(duration.replace('S', ''))
        
        total_seconds = minutes * 60 + seconds
        return total_seconds <= 60
    
    def download_video(self, video_url, video_id):
        """
        Download a video to temp directory.
        
        Args:
            video_url: URL of the video
            video_id: Video ID for filename
            
        Returns:
            Path to downloaded video file
        """
        output_path = self.temp_dir / f"{video_id}.mp4"
        
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': str(output_path),
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            return output_path
        except Exception as e:
            print(f"Error downloading video {video_id}: {e}")
            raise
    
    def analyze_with_gemini(self, video_path, title, views):
        """
        Analyze video with Gemini AI.
        
        Args:
            video_path: Path to video file
            title: Video title
            views: Number of views
            
        Returns:
            Dict with video_description and title_effectiveness_analysis
        """
        prompt = f"""You are analyzing a YouTube Short to understand why its title was effective.

Video Title: "{title}"
Views: {views:,}

Please provide:
1. A detailed description of what happens in this video (2-3 sentences)
2. An analysis of why this title was effective for this content (3-4 sentences covering specific techniques used)

Format your response as:
VIDEO DESCRIPTION:
[Your description here]

TITLE EFFECTIVENESS:
[Your analysis here]"""

        try:
            # Upload video file
            video_file = genai.upload_file(path=str(video_path))
            
            # Wait for video to be processed
            print(f"  Waiting for video processing...")
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                raise ValueError("Video processing failed")
            
            # Generate analysis
            response = self.gemini_model.generate_content([video_file, prompt])
            
            # Parse response
            response_text = response.text
            parts = response_text.split('TITLE EFFECTIVENESS:')
            
            if len(parts) == 2:
                description = parts[0].replace('VIDEO DESCRIPTION:', '').strip()
                effectiveness = parts[1].strip()
            else:
                # Fallback if format isn't followed
                description = response_text[:len(response_text)//2].strip()
                effectiveness = response_text[len(response_text)//2:].strip()
            
            # Clean up uploaded file
            genai.delete_file(video_file.name)
            
            return {
                'video_description': description,
                'title_effectiveness_analysis': effectiveness
            }
            
        except Exception as e:
            print(f"Error analyzing with Gemini: {e}")
            raise
    
    def load_existing_results(self):
        """Load existing results if the output file exists."""
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Return empty structure
        return {
            'metadata': {
                'channel_url': self.channel_url,
                'date_analyzed': datetime.now().strftime('%Y-%m-%d'),
                'total_shorts_analyzed': 0,
                'gemini_model': 'gemini-2.5-flash'
            },
            'shorts': []
        }
    
    def save_results(self, results):
        """Save results to JSON file."""
        results['metadata']['total_shorts_analyzed'] = len(results['shorts'])
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def process_shorts(self):
        """Main processing loop."""
        print("=" * 60)
        print("YouTube Shorts Analyzer")
        print("=" * 60)
        
        # Load existing results
        results = self.load_existing_results()
        analyzed_video_ids = {short['video_id'] for short in results['shorts']}
        
        print(f"Already analyzed: {len(analyzed_video_ids)} shorts")
        
        # Fetch top shorts
        top_shorts = self.fetch_top_shorts()
        
        # Process each short
        for rank, short in enumerate(top_shorts, start=1):
            video_id = short['video_id']
            
            # Skip if already analyzed
            if video_id in analyzed_video_ids:
                print(f"[{rank}/{len(top_shorts)}] Skipping {video_id} (already analyzed)")
                continue
            
            print(f"\n[{rank}/{len(top_shorts)}] Processing: {short['title']}")
            print(f"  Views: {short['views']:,}")
            print(f"  URL: {short['url']}")
            
            try:
                # Download video
                print(f"  Downloading...")
                video_path = self.download_video(short['url'], video_id)
                
                # Analyze with Gemini
                print(f"  Analyzing with Gemini...")
                analysis = self.analyze_with_gemini(
                    video_path,
                    short['title'],
                    short['views']
                )
                
                # Convert duration to seconds
                duration_seconds = self._duration_to_seconds(short['duration'])
                
                # Add to results
                result_entry = {
                    'rank': rank,
                    'video_id': video_id,
                    'url': short['url'],
                    'title': short['title'],
                    'views': short['views'],
                    'published_date': short['published_date'],
                    'duration_seconds': duration_seconds,
                    'gemini_analysis': analysis,
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                results['shorts'].append(result_entry)
                
                # Save incrementally
                self.save_results(results)
                print(f"  ✓ Saved to {self.output_file}")
                
                # Cleanup
                video_path.unlink()
                
            except Exception as e:
                print(f"  ✗ Error processing {video_id}: {e}")
                continue
        
        print("\n" + "=" * 60)
        print(f"Analysis complete! Results saved to {self.output_file}")
        print(f"Total shorts analyzed: {len(results['shorts'])}")
        print("=" * 60)
    
    def _duration_to_seconds(self, duration_str):
        """Convert ISO 8601 duration to seconds."""
        duration = duration_str.replace('PT', '')
        
        minutes = 0
        seconds = 0
        
        if 'M' in duration:
            parts = duration.split('M')
            minutes = int(parts[0])
            if len(parts) > 1 and 'S' in parts[1]:
                seconds = int(parts[1].replace('S', ''))
        elif 'S' in duration:
            seconds = int(duration.replace('S', ''))
        
        return minutes * 60 + seconds
    
    def cleanup(self):
        """Remove temporary download directory."""
        if self.temp_dir.exists():
            for file in self.temp_dir.glob('*'):
                file.unlink()
            self.temp_dir.rmdir()
            print("Cleaned up temporary files")


def main():
    """Main entry point."""
    CHANNEL_URL = "https://www.youtube.com/@PeepingOtter/shorts"
    OUTPUT_FILE = "shorts_analysis.json"
    MAX_SHORTS = 100
    
    analyzer = YouTubeShortAnalyzer(
        channel_url=CHANNEL_URL,
        output_file=OUTPUT_FILE,
        max_shorts=MAX_SHORTS
    )
    
    try:
        analyzer.process_shorts()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Progress has been saved.")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        raise
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()