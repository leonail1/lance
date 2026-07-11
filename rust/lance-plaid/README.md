# lance-plaid

`lance-plaid` contains the CPU-only PLAID storage and search primitives used by
the experimental Lance late-interaction index.

The residual codec, centroid probing, approximate scoring, and MaxSim reranking
designs are derived from NextPlaid commit
`2dbc95f152244a95c8175d163a77a832d8c8c97d`, licensed under Apache-2.0:

<https://github.com/lightonai/next-plaid>

This crate intentionally excludes NextPlaid's SQLite metadata layer, text
encoder, HTTP API, and GPU features. Database filtering and row visibility are
supplied by the embedding database through the `Eligibility` interface.
