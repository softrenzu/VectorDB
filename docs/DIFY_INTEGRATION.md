# Dify integration status

## Current constraint

Dify's VectorDB backend interface is `BaseVector`, with create/add/exists/delete/vector-search/full-text-search/drop operations. The standalone Rooomtech VectorDB server implements those operations.

Dify also keeps the backend type in `VectorType`. Until `rooomtech-vector` is accepted into that enum, a completely new public backend name cannot be selected by an unmodified Dify core.

## No-core-source-change development path

For development, keep Dify source untouched and build a custom API/worker image that installs a Rooomtech VectorDB provider package in place of one unused existing backend provider. The alias is only a temporary deployment bridge; Rooomtech VectorDB does not emulate that database protocol and does not emulate Qdrant.

The adapter source is in `integrations/dify/dify_vector.py`.

## Target path

1. Stabilize the Rooomtech VectorDB server and Python client.
2. Package the Dify adapter as a normal `dify.vector_backends` entry point.
3. Add `ROOOMTECH_VECTOR` to Dify `VectorType` through an upstream pull request.
4. Add official environment/config schema and integration tests.
5. Remove the temporary alias deployment.
