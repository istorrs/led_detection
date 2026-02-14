#!/bin/bash

# Trap SIGINT to exit immediately regarding of where the script is processing
trap "echo 'Script interrupted by user'; exit 1" INT

# Resolve Repo Root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to Repo Root
cd "$REPO_ROOT" || exit 1
echo "Working directory changed to: $(pwd)"

# Default values
SERIAL_PORT="/dev/ttyUSB1"
OUTPUT_DIR="$HOME/Downloads/simulated_aed_pulses"
AED_TYPES_FILE="aed_types.json"
DRY_RUN=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --serial-port) SERIAL_PORT="$2"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        --dry-run) DRY_RUN="--dry-run" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Using Serial Port: $SERIAL_PORT"
echo "Output Directory: $OUTPUT_DIR"
if [ -n "$DRY_RUN" ]; then
    echo "Dry Run Mode: Enabled"
fi

# Check and activate virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "Error: Virtual environment not found in $(pwd)/venv. Please activate it or run from the repo root."
        exit 1
    fi
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Check if aed_types.json exists
if [ ! -f "$AED_TYPES_FILE" ]; then
    # Try finding it relative to repo root if script is run from there
    AED_TYPES_FILE="aed_types.json"
    if [ ! -f "$AED_TYPES_FILE" ]; then
        echo "Error: Could not find aed_types.json"
        exit 1
    fi
fi

# Get list of AED indices
count=$(jq '.aed_types | length' "$AED_TYPES_FILE")

# Generate a unique run ID for this execution (reused across all datasets)
RUN_ID=$(uuidgen | cut -c1-8)
echo "Run ID: $RUN_ID"

for ((i=0; i<$count; i++)); do
    id=$(jq -r ".aed_types[$i].id" "$AED_TYPES_FILE")
    name=$(jq -r ".aed_types[$i].name" "$AED_TYPES_FILE")
    min_frames=$(jq -r ".aed_types[$i].params.blink_frames_min" "$AED_TYPES_FILE")
    max_frames=$(jq -r ".aed_types[$i].params.blink_frames_max" "$AED_TYPES_FILE")
    period_sec=$(jq -r ".aed_types[$i].params.blink_period_seconds" "$AED_TYPES_FILE")

    echo "------------------------------------------------"
    echo "Processing AED: $name"

    # Skip AEDs with "LCD" or "case" in the name (case-insensitive)
    name_lower=$(echo "$name" | tr '[:upper:]' '[:lower:]')
    if [[ "$name_lower" == *"lcd"* ]] || [[ "$name_lower" == *"case"* ]]; then
        echo "  [SKIPPED] Name contains 'LCD' or 'case'"
        continue
    fi

    # Create AED-specific subdirectory
    AED_DIR="$OUTPUT_DIR/$name"
    mkdir -p "$AED_DIR"

    # Calculate Duration (Average of min/max frames at 30FPS)
    # Frame time at 30fps is ~33.333ms
    avg_frames=$(echo "scale=2; ($min_frames + $max_frames) / 2" | bc)
    duration_ms=$(echo "scale=2; $avg_frames * (1000 / 30)" | bc)

    # Calculate Period
    if (( $(echo "$period_sec == -1" | bc -l) )); then
        echo "  [INFO] Aperiodic type, defaulting period to 1000ms for capture"
        period_ms=1000
    else
        period_ms=$(echo "scale=2; $period_sec * 1000" | bc)
    fi

    # Dynamic FPS Detection
    echo "  > Detecting Camera Max FPS..."
    MAX_FPS=$(python3 tests/data_generation/get_max_fps.py)
    echo "    Camera Max FPS: $MAX_FPS"

    # Always use max FPS for capture - Python script decides which datasets to create
    NATIVE_FPS=$MAX_FPS
    echo "    Using Native FPS: $NATIVE_FPS (Python will create 30/25/11 as applicable)"

    # Iterate through Brightness settings only - FPS datasets auto-generated
    for brightness in 1 33 66 100; do
        echo "  > Generating for $NATIVE_FPS FPS (+ 11 FPS derived) at $brightness% Brightness..."

        # The Python script now auto-creates both native FPS and 11 FPS datasets
        case_name="${name}_${NATIVE_FPS}fps_${brightness}pct"

        cmd="python3 tests/data_generation/generate_pulse_library.py \
            --fps $NATIVE_FPS \
            --duration $duration_ms \
            --period $period_ms \
            --brightness $brightness \
            --serial-port $SERIAL_PORT \
            --output-dir \"$AED_DIR\" \
            --case-name \"$case_name\" \
            --aed-type $id \
            --aed-name \"$name\" \
            --color \
            --run-id $RUN_ID \
            $DRY_RUN"

        echo "    Executing: $cmd"
        eval $cmd

        if [ $? -ne 0 ]; then
            echo "    [ERROR] Failed to generate data for $case_name"
        else
            echo "    [SUCCESS] Generated $case_name (and 11fps variant)"
        fi

        # Small sleep between runs
        sleep 1
    done
done

echo "Done."
