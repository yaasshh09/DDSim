---
title: Dropped frames
summary: Why the page can miss some iteration reports while a sweep runs, what that does to the plots, and why the finished curve is still complete.
docs: 02-numerics.md#Convergence criteria
---

## In plain words

While a sweep runs, the solver reports each step it takes: every Newton
iteration or Gummel cycle, every continuation attempt, every finished point.
Those reports, called frames, wait in a queue until the page reads them. If
the page falls behind, the queue fills, and rather than make the solver wait,
the oldest waiting frames are thrown away.

When that happens the note under the curve plot ends with how many frames were
dropped. The residual plot is then missing some iterations. Its horizontal axis
counts the frames that arrived, so the lost ones close up rather than leave a
gap, and a residual can seem to fall faster than it did.

The curve itself is complete. When a job finishes on its own, the page fetches
the finished curve separately and redraws the plot from it, so a point whose
frame was dropped still appears. A job that failed or was cancelled hands back
no finished curve, so there a dropped point frame is simply missing from what
was drawn.

## In more depth

Each job has its own queue, which holds 4096 frames, with one extra slot kept
free for the end of stream marker so that a stream nobody is reading still
closes. A Newton solve sends one frame per iteration and a bias point is tens
of those, so the queue holds minutes of telemetry, and it overflows only when
nothing reads it for a while, for example when the page has closed its
connection.

Only the solver's thread writes to the queue. When it is full, that thread
removes the oldest frame, counts it, and adds the new one. It never blocks, so
the time a solve takes never depends on how fast a browser is. The count
travels in the final status message.

The finished curve never goes through the queue. It is held on the job and
fetched over HTTP once the job reports done, and the profile arrays are fetched
the same way, one point at a time. What a dropped frame can cost is telemetry:
a few lines of the residual plot and a few convergence notes, never a solved
number.
