# Two warmth physics coexist rather than one

The simulator models cold starts under two mutually exclusive rules and every collection is
stamped with which one generated it. Under `platform_reuse_v1` sandbox warmth and image
locality are **coupled** — a platform whose previous task matched the type skips the image
pull entirely. Under `node_disk_v2` they are **decoupled** — only a node-local disk hit
skips the pull, so a warm sandbox never masks an evicted image. We keep both because v1 is
what the earliest corpus was generated under and re-generating it is not worth the compute,
while v2 is the physics we actually believe: image residency is a disk property, and letting
sandbox reuse imply it makes eviction unrepresentable.

## Consequences

This is not a runtime flag with a default — it partitions the data permanently. Collections
generated under different warmth physics can **never** be trained together, which is why the
compatibility matrix splits `legacy_v0_node_disk_v2_4task` (2,816 snapshots, 15 collections)
from `legacy_v0_platform_reuse_v1_4task` (48 snapshots, `regime_b` only). Changing a
collection's physics means regenerating it from scratch.

Note that the two physics differ *only* in whether sandbox warmth can skip an image pull.
Sandbox warmth still gates the cold-start penalty under **both**, so a platform's single
`previous_task` slot is load-bearing either way.
