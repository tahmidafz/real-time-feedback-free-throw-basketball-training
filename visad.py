import asyncio
import cv2
import numpy
import time
import sys
import os
from collections import deque
import traceback
import pyttsx3

# Add the parent directory of 'synchronous' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.server_info import get_ip_and_port
from ganzin.sol_sdk.asynchronous.async_client import AsyncClient, recv_gaze, recv_video
from ganzin.sol_sdk.common_models import Camera
from ganzin.sol_sdk.utils import find_nearest_timestamp_match

# Constants for fixation detection
FIXATION_RADIUS_PX = 35
FIXATION_DURATION_MS = 150
GAZE_HISTORY_MS = 300
AUDIO_FEEDBACK_COOLDOWN = 10  # seconds
EXPERT_HOLD_DURATION = 1.11  # seconds

# Normalized board region (used for gaze visualization and fixation)
# Example with no physical size known, just center and ratio:
frame_width_px = 1328
frame_height_px = 720

# Aspect ratio of your boundaries (example)
rect_w_cm = 10
rect_h_cm = 6.5

rect_aspect = rect_w_cm / rect_h_cm

# To fit it centered horizontally and vertically in normalized coords:
# Choose normalized width (e.g., 0.2)
norm_width = 0.2
norm_height = norm_width / rect_aspect  # maintain aspect ratio

# Center rectangle coordinates in normalized units
x_min = 0.5 - norm_width / 2
x_max = 0.5 + norm_width / 2
y_min = 0.5 - norm_height / 2
y_max = 0.5 + norm_height / 2

board_region = {
    "x_min": x_min,
    "x_max": x_max,
    "y_min": y_min,
    "y_max": y_max
}

def is_inside_board(x, y, width, height, region):
    return (region["x_min"] * width <= x <= region["x_max"] * width) and \
           (region["y_min"] * height <= y <= region["y_max"] * height)

def speak_feedback(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

async def main():
    address, port = get_ip_and_port()
    timeout_seconds = 5.0

    async with AsyncClient(address, port) as ac:
        if not (await ac.get_status()).eye_image_encoding_enabled:
            print('Warning: Please enable eye image encoding and try again.')
            return

        error_event = asyncio.Event()

        frames = asyncio.Queue(1)
        collect_video_task = asyncio.create_task(keep_last_video_frame(ac, frames, error_event))

        gazes = asyncio.Queue()
        collect_gaze_task = asyncio.create_task(collect_gaze(ac, gazes, error_event, timeout_seconds))

        left_eye_data = asyncio.Queue()
        collect_left_eye_task = asyncio.create_task(
            collect_eye(ac, left_eye_data, Camera.LEFT_EYE, error_event, timeout_seconds))

        right_eye_data = asyncio.Queue()
        collect_right_eye_task = asyncio.create_task(
            collect_eye(ac, right_eye_data, Camera.RIGHT_EYE, error_event, timeout_seconds))

        gaze_history = deque()
        hold_start_time = None
        last_feedback_time = 0

        try:
            await present(
                frames, gazes, left_eye_data, right_eye_data,
                error_event, timeout_seconds, gaze_history,
                hold_start_time, last_feedback_time)
        except Exception as e:
            print("An error occurred:", str(e))
            traceback.print_exc()
        finally:
            collect_video_task.cancel()
            collect_gaze_task.cancel()
            collect_left_eye_task.cancel()
            collect_right_eye_task.cancel()

async def keep_last_video_frame(ac: AsyncClient, queue: asyncio.Queue, error_event: asyncio.Event) -> None:
    async for frame in recv_video(ac, Camera.SCENE):
        if error_event.is_set():
            break
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(frame)

async def collect_gaze(ac: AsyncClient, queue: asyncio.Queue, error_event: asyncio.Event, timeout) -> None:
    try:
        async for gaze in recv_gaze(ac):
            if error_event.is_set():
                break
            await asyncio.wait_for(queue.put(gaze), timeout=timeout)
    except Exception:
        error_event.set()

async def collect_eye(ac: AsyncClient, queue: asyncio.Queue, camera: Camera, error_event: asyncio.Event, timeout) -> None:
    try:
        async for eye in recv_video(ac, camera):
            if error_event.is_set():
                break
            await asyncio.wait_for(queue.put(eye), timeout=timeout)
    except Exception:
        error_event.set()

async def present(frames, gaze_queue, left_eye_queue, right_eye_queue, error_event, timeout, gaze_history,
                  hold_start_time, last_feedback_time):
    while not error_event.is_set():
        scene_camera_datum = await get_video_frame(frames, timeout)
        timestamp = scene_camera_datum.get_timestamp()

        gazes = await get_all_queue_items(gaze_queue, timeout)
        gaze = find_nearest_timestamp_match(timestamp, gazes)

        left_eye_data = await get_all_queue_items(left_eye_queue, timeout)
        left_eye = find_nearest_timestamp_match(timestamp, left_eye_data)

        right_eye_data = await get_all_queue_items(right_eye_queue, timeout)
        right_eye = find_nearest_timestamp_match(timestamp, right_eye_data)

        buffer = scene_camera_datum.get_buffer()
        buffer = cv2.resize(buffer, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        height, width, _ = buffer.shape
        now_ms = time.time() * 1000

        # Draw the board region rectangle in green
        cv2.rectangle(
            buffer,
            (int(board_region["x_min"] * width), int(board_region["y_min"] * height)),
            (int(board_region["x_max"] * width), int(board_region["y_max"] * height)),
            (0, 255, 0), 2
        )

        # Extract gaze x,y if valid (validity == 1)
        gaze_x = gaze_y = None
        if gaze is not None:
            try:
                combined = getattr(gaze, "combined", None)
                if combined:
                    gaze_2d = getattr(combined, "gaze_2d", None)
                    if gaze_2d and getattr(gaze_2d, "validity", 0) == 1:
                        gaze_x = int(gaze_2d.x / 2)
                        gaze_y = int(gaze_2d.y / 2)
            except Exception as e:
                print("Error extracting gaze coords:", e)

        # Add current gaze to history if valid
        if gaze_x is not None and gaze_y is not None:
            gaze_history.append((now_ms, gaze_x, gaze_y))

        # Remove old gaze points older than GAZE_HISTORY_MS
        while gaze_history and (now_ms - gaze_history[0][0] > GAZE_HISTORY_MS):
            gaze_history.popleft()

        # Draw recent gaze points in red
        #for _, gx, gy in gaze_history:
        #    cv2.circle(buffer, (gx, gy), 3, (0, 0, 255), -1)

        fixation_detected = False
        fixation_center = (0, 0)
        fixation_duration_ms = 0

        # Fixation detection
        if len(gaze_history) >= 2:
            xs, ys = zip(*[(x, y) for _, x, y in gaze_history])
            center_x = int(sum(xs) / len(xs))
            center_y = int(sum(ys) / len(ys))
            distances = [((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5 for x, y in zip(xs, ys)]
            max_distance = max(distances)
            fixation_duration_ms = gaze_history[-1][0] - gaze_history[0][0]

            if max_distance < FIXATION_RADIUS_PX and fixation_duration_ms >= FIXATION_DURATION_MS:
                fixation_detected = True
                fixation_center = (center_x, center_y)
                # Draw fixation circle (blue)
                cv2.circle(buffer, fixation_center, 12, (255, 0, 0), -1)

                if is_inside_board(center_x, center_y, width, height, board_region):
                    cv2.circle(buffer, fixation_center, 20, (0, 255, 255), 2)

        # Audio feedback logic with hold time and cooldown
        current_time = time.time()
        feedback_text = None

        if fixation_detected:
            # Check if fixation inside board region
            if is_inside_board(fixation_center[0], fixation_center[1], width, height, board_region):
                if hold_start_time is None:
                    hold_start_time = current_time
                    feedback_text = "Please hold on"
                else:
                    hold_duration = current_time - hold_start_time
                    if hold_duration >= EXPERT_HOLD_DURATION:
                        feedback_text = "Great job, keep it up!"
                        hold_start_time = None
                    else:
                        feedback_text = "Please hold on"                   
            else:
                hold_start_time = None
                feedback_text = "Focus on the shooting board!"
        else:
            hold_start_time = None
            feedback_text = "Focus on the shooting board!"

        # Speak feedback with cooldown to avoid spamming
        if feedback_text and (current_time - last_feedback_time) >= AUDIO_FEEDBACK_COOLDOWN:
            print(f"Audio feedback: {feedback_text}")
            # Run speech in separate thread so it won't block asyncio
            asyncio.get_event_loop().run_in_executor(None, speak_feedback, feedback_text)
            last_feedback_time = current_time
        else:
            print(f"[Speech suppressed] {feedback_text}")

        # Draw gaze + eye images on buffer
        draw_gaze(gaze, buffer)
        #draw_to_center_top(buffer, left_eye.get_buffer(), Camera.LEFT_EYE, 0.3)
        #draw_to_center_top(buffer, right_eye.get_buffer(), Camera.RIGHT_EYE, 0.3)

        cv2.imshow('Press "q" to exit', buffer)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

async def get_video_frame(queue, timeout):
    return await asyncio.wait_for(queue.get(), timeout=timeout)

async def get_all_queue_items(queue, timeout):
    items = []

    # Wait for the first item with timeout
    item = await asyncio.wait_for(queue.get(), timeout=timeout)
    items.append(item)

    # Continue retrieving items until the queue is empty
    while True:
        try:
            item = queue.get_nowait()
            items.append(item)
        except asyncio.QueueEmpty:
            break
    return items

def draw_gaze(gaze, frame):
    if gaze is None:
        return frame
    try:
        center = (int(gaze.combined.gaze_2d.x / 2), int(gaze.combined.gaze_2d.y / 2))
        radius = 15
        bgr_color = (255, 255, 0)
        thickness = 3
        cv2.circle(frame, center, radius, bgr_color, thickness)
    except Exception:
        pass
    return frame

def draw_to_center_top(
        scene_cam_frame: numpy.ndarray,
        eye_frame: numpy.ndarray, camera: Camera,
        ratio: float = 1.0,
        center_margin = 5
) -> tuple[int, int]:
    half_frame_width = scene_cam_frame.shape[1] // 2
    pos_x = half_frame_width
    pos_y = 0
    resized_eye_width = int(eye_frame.shape[1] * ratio)
    resized_eye_height = int(eye_frame.shape[0] * ratio)
    resized_eye = cv2.resize(eye_frame, (resized_eye_width, resized_eye_height))

    match camera:
        case Camera.LEFT_EYE:
            pos_x -= (resized_eye_width + center_margin)
        case Camera.RIGHT_EYE:
            pos_x += center_margin
        case _:
            raise ValueError(f"Invalid camera type: {camera}")

    scene_cam_frame[pos_y:pos_y + resized_eye_height,
                    pos_x:pos_x + resized_eye_width] = resized_eye

if __name__ == '__main__':
    asyncio.run(main())
