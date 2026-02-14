#!/usr/bin/env python3
"""
Generate color-swapped LED datasets from blue LED datasets.

IMPORTANT: This script NEVER modifies source data. All operations are on copies.

This script recursively scans a root directory for datasets with .metadata files
containing human-annotated ROI and blink_events. For each frame, it performs a
BGR channel swap within the ROI to convert blue LED pixels to either red or green.

Since a channel swap on dark/off pixels is essentially a no-op, ALL frames are
swapped (not just pulse frames), ensuring consistency across the dataset.

The script:
1. Scans source directory for datasets with .metadata files (READ ONLY)
2. Creates a separate output directory with '_red' or '_green' suffix
3. Copies all non-frame files to the output directory
4. Channel-swaps blue LED pixels in the ROI for every frame (output directory)
5. Source data remains completely untouched

Usage:
    python scripts/generate_blue_led_swaps.py --target-color red [options]

Options:
    --root-dir PATH         Root directory to scan for datasets (default: current working directory)
    --target-color COLOR    Target LED color: 'red' or 'green' (required)
    --dry-run              Show what would be done without making changes
    --verbose              Print detailed progress information
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
import numpy as np

# Import metadata handler for reading .metadata files
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from gui.metadata_handler import load_metadata  # pylint: disable=wrong-import-position,import-error


def scan_folders(root_data_dir: str) -> List[Tuple[str, str]]:
    """
    Scan root directory for datasets with .metadata files containing blink_events and ROI.

    Args:
        root_data_dir: Root directory to scan

    Returns:
        List of (folder_name, full_path) tuples for folders with valid metadata
    """
    cases = []
    for root, dirs, files in os.walk(root_data_dir):  # pylint: disable=unused-variable
        if ".metadata" in files:
            full_path = root
            name = os.path.basename(root)

            # Load metadata to check if blink_events and human_roi exist
            metadata, _ = load_metadata(root, create_default=False)
            if metadata and metadata.get("blink_events") and metadata.get("human_roi"):
                cases.append((name, full_path))

    cases.sort(key=lambda x: x[0])
    return cases


def get_frame_files(
    directory: Path, prefix: str = "LED-frame-", extension: str = ".jpg"
) -> List[str]:
    """
    Get sorted list of frame files in directory.

    Args:
        directory: Directory containing frames
        prefix: Frame filename prefix
        extension: Frame filename extension

    Returns:
        Sorted list of frame filenames
    """
    if not directory.exists():
        return []

    files = []
    for f in directory.iterdir():
        if f.is_file() and f.name.startswith(prefix) and f.name.endswith(extension):
            files.append(f.name)

    # Sort by frame number
    def extract_frame_num(filename: str) -> int:
        name_without_ext = filename.replace(extension, "")
        num_str = name_without_ext.replace(prefix, "")
        return int(num_str)

    files.sort(key=extract_frame_num)
    return files


def filename_to_frame_number(
    filename: str, prefix: str = "LED-frame-", extension: str = ".jpg"
) -> int:
    """
    Convert filename to frame number.

    Args:
        filename: Frame filename
        prefix: Frame filename prefix
        extension: Frame filename extension

    Returns:
        Frame number (0-based index)
    """
    name_without_ext = filename.replace(extension, "")
    num_str = name_without_ext.replace(prefix, "")
    return int(num_str)


def swap_blue_channel(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    target_color: str
) -> Tuple[np.ndarray, int]:
    """
    Swap the blue channel with red or green within the ROI.

    In OpenCV BGR format:
    - Blue->Red: swap channel 0 (B) and channel 2 (R)
    - Blue->Green: swap channel 0 (B) and channel 1 (G)

    Args:
        image: BGR image array
        roi: Region of interest as (x1, y1, x2, y2)
        target_color: Target color ('red' or 'green')

    Returns:
        Tuple of (modified image, number of pixels swapped)
    """
    x1, y1, x2, y2 = roi

    # Clamp ROI to image bounds
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return image, 0

    result = image.copy()
    roi_region = result[y1:y2, x1:x2]

    if target_color == "red":
        # Swap B (channel 0) and R (channel 2)
        roi_region[:, :, 0], roi_region[:, :, 2] = (
            roi_region[:, :, 2].copy(),
            roi_region[:, :, 0].copy(),
        )
    else:
        # Swap B (channel 0) and G (channel 1)
        roi_region[:, :, 0], roi_region[:, :, 1] = (
            roi_region[:, :, 1].copy(),
            roi_region[:, :, 0].copy(),
        )

    pixel_count = (x2 - x1) * (y2 - y1)
    return result, pixel_count


def create_converted_dataset(
    source_dir: Path,
    output_dir: Path,
    roi: Tuple[int, int, int, int],
    target_color: str,
    prefix: str = "LED-frame-",
    extension: str = ".jpg",
    verbose: bool = False
) -> Dict[str, Any]:
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """
    Create a color-swapped copy of a dataset by swapping blue channel in the ROI.

    IMPORTANT: Source directory is NEVER modified. All operations are on copied files.

    Args:
        source_dir: Source dataset directory (READ ONLY)
        output_dir: Output directory for converted dataset
        roi: Region of interest as (x1, y1, x2, y2)
        target_color: Target color ('red' or 'green')
        prefix: Frame filename prefix
        extension: Frame filename extension
        verbose: Print detailed progress

    Returns:
        Dictionary with statistics about the operation
    """
    stats = {
        "frames_swapped": 0,
        "frames_failed": 0,
        "total_pixels_swapped": 0,
    }

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of all frame files
    frame_files = get_frame_files(source_dir, prefix, extension)

    if not frame_files:
        raise ValueError(f"No frame files found in {source_dir}")

    if verbose:
        print(f"  Found {len(frame_files)} frames in source")
        print(f"  ROI: {roi}")
        print(f"  Target color: {target_color}")

    # Process every frame with channel swap
    for frame_file in frame_files:
        frame_num = filename_to_frame_number(frame_file, prefix, extension)
        source_path = source_dir / frame_file
        dest_path = output_dir / frame_file

        img = cv2.imread(str(source_path))
        if img is None:
            if verbose:
                print(f"    Warning: Could not load {frame_file}")
            shutil.copy2(source_path, dest_path)
            stats["frames_failed"] += 1
            continue

        converted_img, pixels = swap_blue_channel(img, roi, target_color)
        cv2.imwrite(str(dest_path), converted_img)
        stats["frames_swapped"] += 1
        stats["total_pixels_swapped"] += pixels

        if verbose:
            print(f"    Frame {frame_num}: swapped {pixels} pixels (blue -> {target_color})")

    # Copy settings.cfg if it exists
    settings_file = source_dir / "settings.cfg"
    if settings_file.exists():
        shutil.copy2(settings_file, output_dir / "settings.cfg")
        if verbose:
            print("  Copied settings.cfg")

    # Copy and update .metadata file
    update_metadata(source_dir, output_dir, target_color, verbose)

    return stats


def update_metadata(source_dir: Path, output_dir: Path, target_color: str, verbose: bool):
    """
    Copy and update the .metadata file with conversion information.

    Args:
        source_dir: Source dataset directory
        output_dir: Output directory for converted dataset
        target_color: Target color ('red' or 'green')
        verbose: Print detailed progress
    """
    metadata_file = source_dir / ".metadata"
    output_metadata_file = output_dir / ".metadata"

    if not metadata_file.exists():
        return

    try:
        with metadata_file.open(encoding="utf-8") as f:
            metadata = json.load(f)

        metadata["note"] = f"{target_color.capitalize()} LED dataset generated from blue LED source"
        metadata["source_led_color"] = "blue"
        metadata["converted_led_color"] = target_color

        with output_metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        if verbose:
            print("  Updated .metadata with conversion note")

    except (json.JSONDecodeError, IOError) as e:
        print(f"  WARNING: Could not process .metadata: {e}")
        shutil.copy2(metadata_file, output_metadata_file)


def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate color-swapped LED datasets from blue LED datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert blue LEDs to red
  python scripts/generate_blue_led_swaps.py --root-dir /path/to/data --target-color red

  # Convert blue LEDs to green
  python scripts/generate_blue_led_swaps.py --root-dir /path/to/data --target-color green

  # Dry run to see what would be done
  python scripts/generate_blue_led_swaps.py --root-dir /path/to/data --target-color red --dry-run --verbose

Output Structure:
  Original:   /path/to/data/dataset-123/
  Red output:   /path/to/data/dataset-123_red/
  Green output: /path/to/data/dataset-123_green/
        """
    )

    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path.cwd(),
        help="Root directory to scan for datasets (default: current directory)"
    )
    parser.add_argument(
        "--target-color",
        type=str,
        required=True,
        choices=["red", "green"],
        help="Target LED color to convert blue to (required)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )

    return parser.parse_args()


def extract_roi(human_roi) -> Tuple[int, int, int, int]:
    """
    Extract ROI coordinates from metadata human_roi field.

    Args:
        human_roi: ROI data as dict with x1/y1/x2/y2 keys or as a sequence

    Returns:
        Tuple of (x1, y1, x2, y2) coordinates

    Raises:
        ValueError: If the ROI format is invalid
    """
    if isinstance(human_roi, dict):
        return (
            int(human_roi["x1"]),
            int(human_roi["y1"]),
            int(human_roi["x2"]),
            int(human_roi["y2"])
        )
    return tuple(int(v) for v in human_roi[:4])


def process_dataset(index, total, name, source_dir, target_color, dry_run, verbose):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    """
    Process a single dataset: extract ROI, create converted copy.

    Args:
        index: Current dataset index (1-based)
        total: Total number of datasets
        name: Dataset folder name
        source_dir: Path to source dataset
        target_color: Target color ('red' or 'green')
        dry_run: If True, skip actual processing
        verbose: Print detailed progress

    Returns:
        Tuple of (success: bool, stats: dict or None, skip_reason: str or None)
    """
    metadata, _ = load_metadata(str(source_dir), create_default=False)
    human_roi = metadata.get("human_roi")

    if not human_roi:
        print(f"\n[{index}/{total}] Skipping {name}: no human_roi in metadata")
        return False, None, "no human_roi"

    try:
        roi = extract_roi(human_roi)
    except (KeyError, TypeError, ValueError) as e:
        print(f"\n[{index}/{total}] Skipping {name}: invalid ROI format: {e}")
        return False, None, f"invalid ROI: {e}"

    output_dir = source_dir.parent / f"{source_dir.name}_{target_color}"

    print(f"\n[{index}/{total}] Processing: {name}")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    print(f"  ROI: {roi}")

    if dry_run:
        print("  (Dry run - skipping actual processing)")
        return True, None, None

    stats = create_converted_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        roi=roi,
        target_color=target_color,
        verbose=verbose
    )

    print(f"  Done: {stats['frames_swapped']} frames swapped, "
          f"{stats['total_pixels_swapped']} pixels per frame")

    return True, stats, None


def print_summary(total_stats, dry_run):
    """
    Print final summary of all processed datasets.

    Args:
        total_stats: Accumulated statistics dictionary
        dry_run: Whether this was a dry run
    """
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Datasets processed:      {total_stats['datasets_processed']}")
    print(f"Datasets skipped:        {total_stats['datasets_skipped']}")
    print(f"Total frames swapped:    {total_stats['total_frames_swapped']}")
    print(f"Total frames failed:     {total_stats['total_frames_failed']}")

    if total_stats["skipped_list"]:
        print("\nSkipped Datasets:")
        for name, reason in total_stats["skipped_list"]:
            print(f"  - {name}: {reason}")

    if dry_run:
        print("\n*** Dry run complete - no files were modified ***")


def main():
    """Main entry point."""
    args = parse_arguments()

    # Verify root directory exists
    if not args.root_dir.exists():
        print(f"Error: Root directory not found: {args.root_dir}")
        return 1

    # Scan for datasets with .metadata files
    print(f"Scanning {args.root_dir} for datasets with .metadata files...")
    datasets = scan_folders(str(args.root_dir))

    if not datasets:
        print(f"No datasets with .metadata (blink_events + human_roi) found in {args.root_dir}")
        return 1

    print(f"Found {len(datasets)} datasets with pulse events and ROI")
    print(f"\n{'='*80}")
    print(f"OPERATION: Copy datasets and convert blue LED to {args.target_color}")
    print(f"{'='*80}")
    print(f"Input root:    {args.root_dir}  (READ ONLY - never modified)")
    print(f"Target color:  {args.target_color}")

    if args.dry_run:
        print(f"\n{'*'*80}")
        print("*** DRY RUN MODE - No files will be created ***")
        print(f"{'*'*80}\n")

    # Process each dataset
    total_stats = {
        "datasets_processed": 0,
        "datasets_skipped": 0,
        "total_frames_swapped": 0,
        "total_frames_failed": 0,
        "skipped_list": []
    }

    for i, (name, full_path) in enumerate(datasets, 1):
        source_dir = Path(full_path)

        try:
            success, stats, skip_reason = process_dataset(
                i, len(datasets), name, source_dir,
                args.target_color, args.dry_run, args.verbose
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"  Error: {e}")
            success = False
            stats = None
            skip_reason = str(e)

        if success and stats:
            total_stats["datasets_processed"] += 1
            total_stats["total_frames_swapped"] += stats["frames_swapped"]
            total_stats["total_frames_failed"] += stats["frames_failed"]
        elif not success:
            total_stats["datasets_skipped"] += 1
            total_stats["skipped_list"].append((name, skip_reason))

    print_summary(total_stats, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
