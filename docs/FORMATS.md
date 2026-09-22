# Binary formats

## Dataset shard: NIYDS1

| Field | Size |
|---|---:|
| magic `NIYDS1\0\0` | 8 |
| version, little-endian u32 | 4 |
| token_count, little-endian u64 | 8 |
| SHA-256 of canonical token bytes | 32 |
| token values, little-endian u32[] | 4 * token_count |

Only token values 0..255 are accepted by the current byte-tokenizer baseline.

## Checkpoint: NIYCK1

| Field | Size |
|---|---:|
| magic `NIYCK1\0\0` | 8 |
| version, little-endian u32 | 4 |
| cursor, little-endian u64 | 8 |
| bound dataset SHA-256 | 32 |
| transitions_seen, little-endian u64 | 8 |
| checkpoint payload SHA-256 | 32 |
| 256 x 256 counts, little-endian u64 | 524288 |

The payload digest covers cursor, dataset identity, transition count, and every model count in canonical order.
