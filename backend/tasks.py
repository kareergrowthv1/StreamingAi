"""
Celery tasks for Streaming backend.
- merge_video_chunks: run FFmpeg via merge_utils, then delete chunks.
"""
from celery import Celery

import config
from merge_utils import do_merge_chunks

app = Celery("streaming", broker=config.REDIS_URL, backend=config.REDIS_URL)
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"


@app.task(bind=True)
def merge_video_chunks(self, client_id: str, position_id: str, candidate_id: str):
    """Enqueue merge: concat chunks with FFmpeg, then delete chunks dir."""
    return do_merge_chunks(client_id, position_id, candidate_id)
