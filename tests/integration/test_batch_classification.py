import json
import pytest
import cv2
import numpy as np
from led_detection.main import PeakMonitor

@pytest.fixture(name="batch_dir")
def fixture_batch_dir(tmp_path):
    """Create a mock directory structure for batch classification."""
    root = tmp_path / "TestData"
    root.mkdir()

    # Case 1: CR2 with a 100ms pulse (3 frames at 30fps)
    cr2_dir = root / "CR2" / "test_run_cr2"
    cr2_dir.mkdir(parents=True)

    # Create 30 frames (1 second)
    # Pulse at frame 10, 11, 12
    for i in range(30):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        if 10 <= i <= 12:
            img.fill(255) # White

        # Filename format: frame_INDEX_TIMESTAMP.jpg
        # TS = i * 0.033
        fname = f"frame_{i}_{i*0.033:.3f}.jpg"
        cv2.imwrite(str(cr2_dir / fname), img)

    # Case 2: Philips with NO pulse (all black)
    phil_dir = root / "Philips" / "test_run_philips"
    phil_dir.mkdir(parents=True)
    for i in range(10):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(phil_dir / f"frame_{i}_{i*0.033:.3f}.jpg"), img)

    # Case 3: Mixed Resolutions (Should filter out bad ones)
    mix_dir = root / "Mixed" / "test_run_mixed"
    mix_dir.mkdir(parents=True)
    # 5 good frames (100x100), 1 bad frame (50x50), 5 good frames
    # Pulse in good frames 4,5,6
    for i in range(11):
        if i == 5:
            img = np.zeros((50, 50, 3), dtype=np.uint8) # Bad size
        else:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            if 4 <= i <= 6:
                img.fill(255)

        cv2.imwrite(str(mix_dir / f"frame_{i}_{i*0.033:.3f}.jpg"), img)

    # Add a template file (should be ignored)
    template_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(mix_dir / "CR2_Template.jpeg"), template_img)

    cv2.imwrite(str(mix_dir / "CR2_Template.jpeg"), template_img)

    # Case 4: Natural Sorting (frame-2 vs frame-10)
    sort_dir = root / "Sort" / "test_run_sort"
    sort_dir.mkdir(parents=True)
    # Create frame-1, frame-2, frame-10.
    # Pulse in frame-10.
    # If sorted alphabetically, frame-10 comes before frame-2.
    # If we parse ID, frame-10 is t=0.333, frame-2 is t=0.066.

    # frame-1
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(sort_dir / "LED-frame-1.jpg"), img)

    # frame-2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(sort_dir / "LED-frame-2.jpg"), img)

    # frame-10 (Pulse)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img.fill(255)
    cv2.imwrite(str(sort_dir / "LED-frame-10.jpg"), img)

    return root

@pytest.mark.integration
def test_batch_classification(batch_dir, tmp_path):
    """Test batch classification logic."""
    output_json = tmp_path / "output.json"

    # Initialize Monitor
    monitor = PeakMonitor(interval=10, threshold=50, use_contrast=False) # default settings

    # Run Batch
    monitor.run_batch_classification(str(batch_dir), str(output_json))

    # Verify Output
    assert output_json.exists()

    with open(output_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Needs to match 2 folders (one might be skipped if no signal, let's check logs)
    # The code says if not signal_values: continue.
    # For Black images, variance is 0.
    # _find_roi_from_variance logs warning if max_val < 10.0 and returns None?
    # No, currently analyze_frame_buffer calls _find_roi_from_variance.
    # _find_roi_from_variance logs warning and returns None if max_val < 10.0.
    # analyze_frame_buffer returns None, [], [].
    # run_batch sees signal_values as empty and continues.

    # We expect CR2, Mixed, and Sort entries.
    assert len(data) == 3

    # Sort by name to be sure
    data.sort(key=lambda x: x['name'])

    # CR2
    cr2_entry = next(d for d in data if d['name'] == 'test_run_cr2')
    assert cr2_entry['aed_type'] == 'CR2'
    assert cr2_entry['expected_count'] == 1
    assert 9 <= cr2_entry['expected_frames'][0] <= 11

    # Mixed
    mixed_entry = next(d for d in data if d['name'] == 'test_run_mixed')
    assert mixed_entry['expected_count'] == 1
    # Pulse frames 4,5,6 -> start index ~4
    assert 3 <= mixed_entry['expected_frames'][0] <= 5

    # Sort
    sort_entry = next(d for d in data if d['name'] == 'test_run_sort')
    assert sort_entry['expected_count'] == 1
    # Pulse is in the 3rd frame loaded (frame-10), so index should be 2
    # Timestamps: frame-1(0.033), frame-2(0.066), frame-10(0.333)
    # List is sorted by time.
    # With Off-By-One fix, we report i-1 (the last low frame)
    assert sort_entry['expected_frames'][0] == 1

    # Verify Videos are NOT created by default
    assert not (batch_dir / "CR2" / "test_run_cr2" / "classification_preview_test_run_cr2.mp4").exists()

    # --- Run 2: With Video Generation ---
    video_output_json = tmp_path / "video_output.json"
    monitor.run_batch_classification(str(batch_dir), str(video_output_json), save_videos=True)

    # Verify Videos ARE created
    assert (batch_dir / "CR2" / "test_run_cr2" / "classification_preview_test_run_cr2.mp4").exists()
    assert (batch_dir / "Philips" / "test_run_philips" / "classification_preview_test_run_philips.mp4").exists()

    # Check file size > 0
    vid_path = batch_dir / "CR2" / "test_run_cr2" / "classification_preview_test_run_cr2.mp4"
    assert vid_path.stat().st_size > 0
