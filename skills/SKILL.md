# unitorch-microsoft-skills

This document describes how to call the apis exposed by the unitorch_microsoft as agent skills.
All examples use the Python wrappers in `unitorch_microsoft.apps.skills.spaces` and `unitorch_microsoft.apps.skills.studios`. 

```python
from unitorch_microsoft.apps.skills.spaces  import SpacesClient
from unitorch_microsoft.apps.skills.studios import StudiosClient

spaces  = SpacesClient("http://127.0.0.1:5000")
studios = StudiosClient("http://127.0.0.1:5000")
```

---

## spaces.md → [Spaces Skills](apps/spaces.md)

Covers: image classification, quality scoring, object detection, image generation/editing.

## studios.md → [Studios Skills](apps/studios.md)

**Important** Use this skill to get the details about the dataset/job/label task/report by the entity id.

Covers: chat sessions, datasets, jobs, label tasks, reports, file uploads.
