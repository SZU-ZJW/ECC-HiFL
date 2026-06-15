"""ECR-HiFL repair subpackage: localization -> patch -> tests.

Both modules are runnable-when-resourced: ``patch_generator`` needs the generation server
(``EP.generation``); ``test_runner`` needs the SWE-bench harness + Docker. Offline they no-op
with a clear message rather than crashing the pipeline.
"""
