import os
import sys
import argparse
import subprocess

try:
    from datasets import load_dataset
    import yt_dlp
    import pandas as pd
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Please install them using:")
    print("pip install datasets yt-dlp pandas tqdm")
    sys.exit(1)

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def reencode_to_h264(input_path, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-strict", "-2",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def main():
    parser = argparse.ArgumentParser(description="Prepare ActivityNet Captions subset for Experiment 0.")
    parser.add_argument("--out_dir", type=str, default="data/activitynet_exp0", help="Output directory")
    parser.add_argument("--max_videos", type=int, default=5, help="Number of videos to download successfully")
    parser.add_argument("--max_events_per_video", type=int, default=3, help="Max events to extract per video")
    parser.add_argument("--reencode_h264", action="store_true", help="Reencode downloaded videos to H.264 MP4")
    args = parser.parse_args()

    # Enforce ffmpeg check if reencode is requested
    if args.reencode_h264 and not check_ffmpeg():
        print("ffmpeg is required for --reencode_h264")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    videos_dir = os.path.join(args.out_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    print("Loading ActivityNet Captions dataset...")
    dataset = None
    # Try Leyo first
    try:
        dataset = load_dataset("Leyo/ActivityNet_Captions", split="train")
    except Exception as e:
        print(f"Failed to load 'Leyo/ActivityNet_Captions': {e}. Trying fallback 'friedrichor/ActivityNet_Captions'...")
        try:
            dataset = load_dataset("friedrichor/ActivityNet_Captions", split="train")
        except Exception as fallback_e:
            print(f"Failed to load fallback dataset: {fallback_e}")
            sys.exit(1)

    if not dataset:
        print("Dataset not loaded successfully.")
        sys.exit(1)

    print(f"Dataset loaded. Total rows: {len(dataset)}")
    
    downloaded_videos = []
    skipped_videos = {}

    ydl_opts = {
        'format': 'best[ext=mp4][height<=360]/best[height<=360]/best[ext=mp4]/best',
        'outtmpl': os.path.join(videos_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }

    # Iterate through the dataset
    pbar = tqdm(total=args.max_videos, desc="Downloading videos")
    for row in dataset:
        if len(downloaded_videos) >= args.max_videos:
            break

        raw_id = row.get("video_id", "")
        # ActivityNet IDs often start with 'v_' in some versions
        video_id = raw_id[2:] if raw_id.startswith("v_") else raw_id
        if not video_id:
            continue

        if video_id in skipped_videos or any(v["video_id"] == video_id for v in downloaded_videos):
            continue

        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Download using yt-dlp
        temp_out = os.path.join(videos_dir, f"{video_id}_temp.mp4")
        final_out = os.path.join(videos_dir, f"{video_id}.mp4")

        # Check if already downloaded/exists
        if os.path.exists(final_out):
            print(f"Video {video_id} already exists. Skipping download.")
            downloaded_videos.append({
                "video_id": video_id,
                "path": final_out,
                "row": row
            })
            pbar.update(1)
            continue

        print(f"\nTrying to download {video_id} from {url}...")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_file = ydl.prepare_filename(info)
                
            if not os.path.exists(downloaded_file):
                # Search for any file matching video_id in output directory
                matches = [f for f in os.listdir(videos_dir) if f.startswith(video_id)]
                if matches:
                    downloaded_file = os.path.join(videos_dir, matches[0])
                else:
                    raise FileNotFoundError("Could not find downloaded file.")

            # Reencode if needed
            if args.reencode_h264:
                print(f"Reencoding {downloaded_file} to H.264...")
                reencode_to_h264(downloaded_file, final_out)
                if downloaded_file != final_out and os.path.exists(downloaded_file):
                    os.remove(downloaded_file)
            else:
                if downloaded_file != final_out:
                    os.rename(downloaded_file, final_out)
            
            downloaded_videos.append({
                "video_id": video_id,
                "path": final_out,
                "row": row
            })
            pbar.update(1)
        except Exception as err:
            reason = str(err).split("\n")[0]
            print(f"Failed to download {video_id}: {reason}")
            skipped_videos[video_id] = reason

    pbar.close()

    # Now write the CSV
    csv_rows = []
    for item in downloaded_videos:
        vid = item["video_id"]
        vpath = item["path"]
        row = item["row"]
        duration = row.get("duration", 0.0)
        
        # Get events
        # We need event_label (caption), start_time, end_time
        # timestamps is a list of lists of floats [start, end]
        # sentences is a list of strings
        timestamps = row.get("timestamps", [])
        sentences = row.get("sentences", [])
        
        # Limit events
        num_events = min(len(timestamps), args.max_events_per_video)
        for i in range(num_events):
            start_t, end_t = timestamps[i]
            caption = sentences[i]
            query = f"Find the event: {caption}"
            
            csv_rows.append({
                "video_id": vid,
                "path": os.path.abspath(vpath),
                "query": query,
                "start_time": start_t,
                "end_time": end_t,
                "event_label": caption,
                "split": "train", # all from train split here
                "duration": duration
            })

    csv_path = os.path.join(args.out_dir, "activitynet_exp0_subset.csv")
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path} with {len(df)} event rows.")

    # Log skipped videos summary
    if skipped_videos:
        print("\nSkipped videos list:")
        for vid, reason in skipped_videos.items():
            print(f"  - {vid}: {reason}")

if __name__ == "__main__":
    main()
