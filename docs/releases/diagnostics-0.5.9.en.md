# mindie-diagnostics 0.5.9: retain child exception evidence

[中文](diagnostics-0.5.9.md)

Publishes merged #37 with a shared, bounded exception-evidence API for owned standalone plugin scripts and child-process transport. Evidence retains owned module/function/line, exception class chains, a recomputed stack fingerprint and typed errno. Generic parent wrappers no longer replace more specific child evidence.

Messages, local paths, locals, source text, stdout, stderr and arbitrary exception attributes are excluded. Transported data is revalidated. Existing reporting consent, non-backfill and incident deduplication rules remain unchanged.

This improves future diagnostics; it does not identify or repair the historical trigger in mindie-agent#324. Existing installations must separately discover, prepare and activate the component update. Already-loaded host processes require a natural reload.

The formal release workflow validates the full component suite, exact-tag benchmarks, distribution files and SHA256SUMS and attaches its evidence to the Release. Original-incident reproduction and NPU acceptance are not claimed.
