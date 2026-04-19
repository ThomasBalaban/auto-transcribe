"""
Analyze Top YouTube Shorts (standalone analysis tool)

Fetches a channel's top shorts, downloads them, runs Gemini analysis, and
saves the result to JSON. Previously used to build training data for the
in-pipeline title generator; title generation has since moved to a
different script, but this tool is kept for ad-hoc channel analysis.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Third-party imports
try:
    from googleapiclient.discovery import build  # type: ignore
    from googleapiclient.errors import HttpError  # type: ignore
    import yt_dlp  # type: ignore
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install google-api-python-client yt-dlp")
    sys.exit(1)

from google.genai import types  # type: ignore

from utils.models import (
    MODEL_FLASH,
    get_gemini_client,
    get_safety_settings,
)


class YouTubeShortAnalyzer:
    """Handles fetching, downloading, and analyzing YouTube shorts."""

    def __init__(self, channel_url, output_file="shorts_analysis.json",
                 max_shorts=100):
        self.channel_url = channel_url
        self.output_file = output_file
        self.max_shorts = max_shorts
        self.temp_dir = Path("temp_short_download")
        self.temp_dir.mkdir(exist_ok=True)

        # YouTube Data API
        self.youtube_api_key = self._load_youtube_api_key()
        self.youtube = build(
            'youtube', 'v3', developerKey=self.youtube_api_key)

        # Gemini client (new SDK)
        self.gemini_client = get_gemini_client()
        self.gemini_safety = get_safety_settings()

        self.channel_handle = self._extract_channel_handle(channel_url)

    def _load_youtube_api_key(self):
        """Load YouTube API key from config.json"""
        config_path = Path(__file__).parent / 'utils' / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(
                "config.json not found. Please add YOUTUBE_API_KEY to config.")

        with open(config_path, 'r') as f:
            config = json.load(f)

        api_key = config.get("YOUTUBE_API_KEY")
        if not api_key:
            raise ValueError("YOUTUBE_API_KEY not found in config.json")

        return api_key

    def _extract_channel_handle(self, url):
        parts = url.rstrip('/').split('/')
        for part in parts:
            if part.startswith('@'):
                return part
        raise ValueError(
            f"Could not extract channel handle from URL: {url}")

    def fetch_top_shorts(self):
        """Fetch top shorts from the channel sorted by view count."""
        print(f"Fetching shorts from channel: {self.channel_handle}")

        try:
            channel_response = self.youtube.channels().list(
                part='id,contentDetails',
                forHandle=self.channel_handle.lstrip('@'),
                maxResults=1,
            ).execute()

            if not channel_response.get('items'):
                raise ValueError(f"Channel not found: {self.channel_handle}")

            channel_id = channel_response['items'][0]['id']
            print(f"Found channel ID: {channel_id}")

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
                    order='date',
                ).execute()

                video_ids = [
                    item['id']['videoId']
                    for item in search_response.get('items', [])
                ]
                if not video_ids:
                    break

                all_videos.extend(video_ids)
                print(
                    f"  Page {page_count}: Found {len(video_ids)} videos "
                    f"(total: {len(all_videos)})"
                )

                next_page_token = search_response.get('nextPageToken')
                if not next_page_token:
                    break

            print(f"Total videos found: {len(all_videos)}")

            print("Filtering for shorts and getting view counts...")
            shorts = []

            for i in range(0, len(all_videos), 50):
                batch = all_videos[i:i + 50]

                videos_response = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    id=','.join(batch),
                ).execute()

                for video in videos_response.get('items', []):
                    duration = video['contentDetails']['duration']
                    if self._is_short_duration(duration):
                        shorts.append({
                            'video_id': video['id'],
                            'title': video['snippet']['title'],
                            'views': int(
                                video['statistics'].get('viewCount', 0)),
                            'published_date': (
                                video['snippet']['publishedAt'].split('T')[0]
                            ),
                            'duration': duration,
                            'url': (
                                f"https://www.youtube.com/shorts/"
                                f"{video['id']}"
                            ),
                        })

                print(
                    f"  Processed {min(i+50, len(all_videos))}/"
                    f"{len(all_videos)} videos, "
                    f"found {len(shorts)} shorts so far"
                )

            print(f"\nTotal shorts found: {len(shorts)}")

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
        duration = duration_str.replace('PT', '')
        if 'H' in duration:
            return False

        minutes = 0
        seconds = 0
        if 'M' in duration:
            parts = duration.split('M')
            minutes = int(parts[0])
            if len(parts) > 1 and 'S' in parts[1]:
                seconds = int(parts[1].replace('S', ''))
        elif 'S' in duration:
            seconds = int(duration.replace('S', ''))

        return (minutes * 60 + seconds) <= 60

    def download_video(self, video_url, video_id):
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
        """Analyze video with Gemini via the new SDK."""
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

        video_file = None
        try:
            # Explicit MIME type — required for reliable File API usage.
            ext = Path(video_path).suffix.lower()
            mime_map = {
                '.mp4': 'video/mp4',
                '.mov': 'video/quicktime',
                '.mkv': 'video/x-matroska',
                '.webm': 'video/webm',
                '.avi': 'video/x-msvideo',
            }
            mime_type = mime_map.get(ext, 'video/mp4')

            video_file = self.gemini_client.files.upload(
                file=str(video_path),
                config={"mime_type": mime_type},
            )

            # Wait for ACTIVE state
            print("  Waiting for video processing...")
            poll_start = time.time()
            while True:
                state = video_file.state.name
                if state == "ACTIVE":
                    break
                if state == "FAILED":
                    raise ValueError("Video processing failed")
                if time.time() - poll_start > 300:
                    raise TimeoutError(
                        f"File did not become ACTIVE within 5 min "
                        f"(last state: {state})"
                    )
                time.sleep(2)
                video_file = self.gemini_client.files.get(
                    name=video_file.name)

            file_part = types.Part.from_uri(
                file_uri=video_file.uri,
                mime_type=video_file.mime_type or mime_type,
            )

            response = self.gemini_client.models.generate_content(
                model=MODEL_FLASH,
                contents=[file_part, prompt],
                config=types.GenerateContentConfig(
                    safety_settings=self.gemini_safety,
                ),
            )

            response_text = response.text
            parts = response_text.split('TITLE EFFECTIVENESS:')

            if len(parts) == 2:
                description = parts[0].replace(
                    'VIDEO DESCRIPTION:', '').strip()
                effectiveness = parts[1].strip()
            else:
                description = response_text[:len(response_text) // 2].strip()
                effectiveness = response_text[len(response_text) // 2:].strip()

            return {
                'video_description': description,
                'title_effectiveness_analysis': effectiveness,
            }

        except Exception as e:
            print(f"Error analyzing with Gemini: {e}")
            raise
        finally:
            if video_file is not None:
                try:
                    self.gemini_client.files.delete(name=video_file.name)
                except Exception:
                    pass

    def load_existing_results(self):
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        return {
            'metadata': {
                'channel_url': self.channel_url,
                'date_analyzed': datetime.now().strftime('%Y-%m-%d'),
                'total_shorts_analyzed': 0,
                'gemini_model': MODEL_FLASH,
            },
            'shorts': [],
        }

    def save_results(self, results):
        results['metadata']['total_shorts_analyzed'] = len(results['shorts'])
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def process_shorts(self):
        print("=" * 60)
        print("YouTube Shorts Analyzer")
        print("=" * 60)

        results = self.load_existing_results()
        analyzed_video_ids = {
            short['video_id'] for short in results['shorts']
        }

        print(f"Already analyzed: {len(analyzed_video_ids)} shorts")
        top_shorts = self.fetch_top_shorts()

        for rank, short in enumerate(top_shorts, start=1):
            video_id = short['video_id']
            if video_id in analyzed_video_ids:
                print(
                    f"[{rank}/{len(top_shorts)}] Skipping {video_id} "
                    f"(already analyzed)"
                )
                continue

            print(
                f"\n[{rank}/{len(top_shorts)}] Processing: {short['title']}")
            print(f"  Views: {short['views']:,}")
            print(f"  URL: {short['url']}")

            try:
                print("  Downloading...")
                video_path = self.download_video(short['url'], video_id)

                print("  Analyzing with Gemini...")
                analysis = self.analyze_with_gemini(
                    video_path, short['title'], short['views'])

                duration_seconds = self._duration_to_seconds(short['duration'])

                result_entry = {
                    'rank': rank,
                    'video_id': video_id,
                    'url': short['url'],
                    'title': short['title'],
                    'views': short['views'],
                    'published_date': short['published_date'],
                    'duration_seconds': duration_seconds,
                    'gemini_analysis': analysis,
                    'analysis_timestamp': datetime.now().isoformat(),
                }

                results['shorts'].append(result_entry)
                self.save_results(results)
                print(f"  ✓ Saved to {self.output_file}")

                video_path.unlink()

            except Exception as e:
                print(f"  ✗ Error processing {video_id}: {e}")
                continue

        print("\n" + "=" * 60)
        print(f"Analysis complete! Results saved to {self.output_file}")
        print(f"Total shorts analyzed: {len(results['shorts'])}")
        print("=" * 60)

    def _duration_to_seconds(self, duration_str):
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
        if self.temp_dir.exists():
            for file in self.temp_dir.glob('*'):
                file.unlink()
            self.temp_dir.rmdir()
            print("Cleaned up temporary files")


def main():
    CHANNEL_URL = "https://www.youtube.com/@PeepingOtter/shorts"
    OUTPUT_FILE = "shorts_analysis.json"
    MAX_SHORTS = 100

    analyzer = YouTubeShortAnalyzer(
        channel_url=CHANNEL_URL,
        output_file=OUTPUT_FILE,
        max_shorts=MAX_SHORTS,
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