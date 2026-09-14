"""Source-aware content authoring — bridges source reading and document generation.

An LLM-backed author reads actual source material, interprets the task,
and produces structured Document-compatible content with provenance and
explicit unresolved questions.  The output feeds directly into the
existing `documents` renderers (DOCX/PPTX/PDF).
"""
